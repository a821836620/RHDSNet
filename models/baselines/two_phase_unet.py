import torch
import torch.nn as nn

from models.segmentation_unet3d import SegmentationUNet3D


class TwoPhaseUNetProxy(nn.Module):
    """Tumor-sen-like two-phase proxy model.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        self.net = SegmentationUNet3D(2, 1, int(model_cfg.get("base_channels", 24)), int(model_cfg.get("levels", 4)))

    def forward(self, x):
        if isinstance(x, dict):
            v0 = x["v0"]
            v1 = x["post"][:, :1] if "post" in x else v0
            x = torch.cat([v0, v1], dim=1)
        if x.shape[1] == 1:
            x = torch.cat([x, x], dim=1)
        return {"logits": self.net(x[:, :2])}


class ALMNProxy(nn.Module):
    """ALMN-like two-phase proxy with residual alignment.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        base = int(model_cfg.get("base_channels", 24))
        self.align = nn.Sequential(nn.Conv3d(2, base, 3, padding=1), nn.SiLU(inplace=True), nn.Conv3d(base, 2, 1))
        self.net = SegmentationUNet3D(2, 1, base, int(model_cfg.get("levels", 4)))

    def forward(self, x):
        if isinstance(x, dict):
            v0 = x["v0"]
            v1 = x["post"][:, :1] if "post" in x else v0
            x = torch.cat([v0, v1 - v0], dim=1)
        if x.shape[1] == 1:
            x = torch.cat([x, torch.zeros_like(x)], dim=1)
        aligned = x[:, :2] + self.align(x[:, :2])
        return {"logits": self.net(aligned)}
