import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal diffusion-step embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeConvBlock(nn.Module):
    """3D Conv block modulated by a diffusion-step embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Conv3d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)
        h = h + self.time_proj(t_emb)[:, :, None, None, None]
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return self.act(h + self.res(x))


class DiffusionUNet3D(nn.Module):
    """3D U-Net denoiser for recurrent conditional diffusion.

    Input:
      z_s: noisy target phase, [B, 1, D, H, W]
      context: previous context phase, [B, 1, D, H, W]
      t: diffusion step, [B]

    Output:
      epsilon_pred: [B, 1, D, H, W]
      encoder_features: list of L tensors [B, C_l, D_l, H_l, W_l]
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 1, base_channels: int = 24, levels: int = 4):
        super().__init__()
        self.levels = int(levels)
        self.channels = [base_channels * (2**i) for i in range(self.levels)]
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(inplace=True),
            nn.Linear(time_dim, time_dim),
        )

        enc = []
        prev = in_channels
        for ch in self.channels:
            enc.append(TimeConvBlock(prev, ch, time_dim))
            prev = ch
        self.encoders = nn.ModuleList(enc)
        self.down = nn.MaxPool3d(2)

        upconvs, decoders = [], []
        for idx in range(self.levels - 1, 0, -1):
            upconvs.append(nn.ConvTranspose3d(self.channels[idx], self.channels[idx - 1], kernel_size=2, stride=2))
            decoders.append(TimeConvBlock(self.channels[idx - 1] * 2, self.channels[idx - 1], time_dim))
        self.upconvs = nn.ModuleList(upconvs)
        self.decoders = nn.ModuleList(decoders)
        self.out = nn.Conv3d(self.channels[0], out_channels, kernel_size=1)

    def _apply_feature_gate(self, features: List[torch.Tensor], gate: Optional[torch.Tensor]) -> List[torch.Tensor]:
        if gate is None:
            return features
        gated = []
        for feat in features:
            g = F.interpolate(gate, size=feat.shape[-3:], mode="trilinear", align_corners=False)
            gated.append(feat * g)
        return gated

    def forward(
        self,
        z_s: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor,
        feature_gate: Optional[torch.Tensor] = None,
        return_features: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        x = torch.cat([z_s, context], dim=1)  # [B, 2, D, H, W]
        t_emb = self.time_mlp(t)
        skips = []
        h = x
        for level, encoder in enumerate(self.encoders):
            h = encoder(h, t_emb)
            skips.append(h)
            if level != self.levels - 1:
                h = self.down(h)

        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips[:-1])):
            h = up(h)
            if h.shape[-3:] != skip.shape[-3:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = decoder(torch.cat([h, skip], dim=1), t_emb)
        eps = self.out(h)
        if return_features:
            return eps, self._apply_feature_gate(skips, feature_gate)
        return eps, []
