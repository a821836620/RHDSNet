from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .diffusion_unet3d import _norm_groups


class ConvBlock3D(nn.Module):
    """Residual 3D Conv block for segmentation."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(_norm_groups(out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
        self.res = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.res(x))


class SegmentationUNet3D(nn.Module):
    """3D U-Net segmentation backbone.

    Input V0 shape: [B, C, D, H, W]
    Encoder features: F_S = {sf^l}, l=1..L
    Decoder output logits: [B, 1, D, H, W]
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 24, levels: int = 4):
        super().__init__()
        self.levels = int(levels)
        self.channels = [base_channels * (2**i) for i in range(self.levels)]
        encoders = []
        prev = in_channels
        for ch in self.channels:
            encoders.append(ConvBlock3D(prev, ch))
            prev = ch
        self.encoders = nn.ModuleList(encoders)
        self.down = nn.MaxPool3d(2)

        upconvs, decoders = [], []
        for idx in range(self.levels - 1, 0, -1):
            upconvs.append(nn.ConvTranspose3d(self.channels[idx], self.channels[idx - 1], 2, stride=2))
            decoders.append(ConvBlock3D(self.channels[idx - 1] * 2, self.channels[idx - 1]))
        self.upconvs = nn.ModuleList(upconvs)
        self.decoders = nn.ModuleList(decoders)
        self.out = nn.Conv3d(self.channels[0], out_channels, 1)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = []
        h = x
        for level, encoder in enumerate(self.encoders):
            h = encoder(h)
            feats.append(h)
            if level != self.levels - 1:
                h = self.down(h)
        return feats

    def decode(self, features: List[torch.Tensor]) -> torch.Tensor:
        h = features[-1]
        for up, decoder, skip in zip(self.upconvs, self.decoders, reversed(features[:-1])):
            h = up(h)
            if h.shape[-3:] != skip.shape[-3:]:
                h = F.interpolate(h, size=skip.shape[-3:], mode="trilinear", align_corners=False)
            h = decoder(torch.cat([h, skip], dim=1))
        return self.out(h)

    def forward(self, x: torch.Tensor, fused_features: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        features = self.encode(x)
        if fused_features is not None:
            features = fused_features
        return self.decode(features)
