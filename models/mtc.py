import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MTCBlock(nn.Module):
    """Pooled key/value cross-attention for one feature scale."""

    def __init__(self, channels: int, attn_channels: int = 16, kv_pool_size=(2, 4, 4), chunk_size: int = 32768):
        super().__init__()
        self.q = nn.Conv3d(channels, attn_channels, 1)
        self.k = nn.Conv3d(channels, attn_channels, 1)
        self.v = nn.Conv3d(channels, attn_channels, 1)
        self.out = nn.Conv3d(attn_channels, channels, 1)
        self.kv_pool_size = tuple(kv_pool_size)
        self.chunk_size = int(chunk_size)
        self.scale = math.sqrt(attn_channels)

    def _attend_one_phase(self, sf: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        b, _, d, h, w = sf.shape
        q = self.q(sf).flatten(2).transpose(1, 2)  # [B, N, Cq]
        hf_pool = F.adaptive_avg_pool3d(hf, self.kv_pool_size)
        k = self.k(hf_pool).flatten(2)  # [B, Cq, M]
        v = self.v(hf_pool).flatten(2).transpose(1, 2)  # [B, M, Cq]
        chunks = []
        for start in range(0, q.shape[1], self.chunk_size):
            q_chunk = q[:, start : start + self.chunk_size]
            attn = torch.softmax(torch.bmm(q_chunk, k) / self.scale, dim=-1)
            chunks.append(torch.bmm(attn, v))
        ctx = torch.cat(chunks, dim=1).transpose(1, 2).reshape(b, -1, d, h, w)
        return self.out(ctx)

    def forward(self, sf: torch.Tensor, hfs_for_scale: List[torch.Tensor]) -> torch.Tensor:
        if not hfs_for_scale:
            return sf
        ctx = 0.0
        for hf in hfs_for_scale:
            if hf.shape[-3:] != sf.shape[-3:]:
                hf = F.interpolate(hf, size=sf.shape[-3:], mode="trilinear", align_corners=False)
            ctx = ctx + self._attend_one_phase(sf, hf)
        return sf + ctx / float(len(hfs_for_scale))


class MultiScaleTaskAlignedConditioning(nn.Module):
    """MTC: use spatial features as query and RHDD hemodynamic features as key/value."""

    def __init__(
        self,
        channels: Sequence[int],
        inject_layers: Sequence[int] = (1, 2, 3, 4),
        attn_channels: int = 16,
        kv_pool_size=(2, 4, 4),
        enabled: bool = True,
        chunk_size: int = 32768,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.inject_layers = {int(x) for x in inject_layers}
        self.blocks = nn.ModuleList(
            [MTCBlock(ch, attn_channels, kv_pool_size, chunk_size) for ch in channels]
        )

    def forward(self, spatial_features: List[torch.Tensor], hemo_features: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Fuse multi-phase RHDD features.

        hemo_features is indexed as hemo_features[phase][level].
        Layer ids in config are 1-based: hf1, hf2, hf3, hf4.
        """
        if not self.enabled or not hemo_features:
            return spatial_features
        fused = []
        for level, sf in enumerate(spatial_features, start=1):
            if level not in self.inject_layers:
                fused.append(sf)
                continue
            hfs_for_scale = [phase_feats[level - 1] for phase_feats in hemo_features if len(phase_feats) >= level]
            fused.append(self.blocks[level - 1](sf, hfs_for_scale))
        return fused
