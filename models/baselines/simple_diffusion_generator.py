import torch
import torch.nn as nn

from models.segmentation_unet3d import ConvBlock3D


class SimplePhaseGenerator(nn.Module):
    """Lightweight V0-to-DCE generator used by generate-then-segment proxies."""

    def __init__(self, num_post_phases: int = 4, base_channels: int = 24):
        super().__init__()
        self.num_post_phases = int(num_post_phases)
        self.net = nn.Sequential(
            ConvBlock3D(1, base_channels),
            ConvBlock3D(base_channels, base_channels),
            nn.Conv3d(base_channels, self.num_post_phases, 1),
        )

    def forward(self, v0):
        residual = self.net(v0)
        return v0.repeat(1, self.num_post_phases, 1, 1, 1) + residual


class DDPMGeneratorProxy(SimplePhaseGenerator):
    """DDPM generator proxy.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation.
    """
