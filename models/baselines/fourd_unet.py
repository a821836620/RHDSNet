import torch
import torch.nn as nn

from models.segmentation_unet3d import SegmentationUNet3D


class FourDUNetProxy(nn.Module):
    """TSESNet/PLHN-like 4D temporal fusion proxy.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation. DCE phases are represented as channels and
    fused by learned 1x1x1 temporal mixing before a 3D U-Net.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        self.num_channels = int(model_cfg.get("num_post_phases", 4)) + 1
        base = int(model_cfg.get("base_channels", 24))
        self.temporal = nn.Sequential(nn.Conv3d(self.num_channels, base, 1), nn.SiLU(inplace=True), nn.Conv3d(base, 1, 1))
        self.net = SegmentationUNet3D(1, 1, base, int(model_cfg.get("levels", 4)))

    def forward(self, x):
        if isinstance(x, dict):
            x = torch.cat([x["v0"], x["post"]], dim=1) if "post" in x else x["v0"]
        if x.shape[1] < self.num_channels:
            pad = x[:, -1:].repeat(1, self.num_channels - x.shape[1], 1, 1, 1)
            x = torch.cat([x, pad], dim=1)
        x = self.temporal(x[:, : self.num_channels])
        return {"logits": self.net(x)}


class TemporalFusionProxy(nn.Module):
    """DEFNet/AKF-Net/BBDiffSF-like dual/kinetic fusion proxy.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation. It fuses anatomical V0 and kinetic residuals
    through two light encoders before segmentation.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        t = int(model_cfg.get("num_post_phases", 4))
        base = int(model_cfg.get("base_channels", 24))
        self.t = t
        self.anat = nn.Sequential(nn.Conv3d(1, base, 3, padding=1), nn.GroupNorm(4 if base % 4 == 0 else 1, base), nn.SiLU())
        self.kin = nn.Sequential(nn.Conv3d(t, base, 3, padding=1), nn.GroupNorm(4 if base % 4 == 0 else 1, base), nn.SiLU())
        self.mix = nn.Conv3d(base * 2, 1, 1)
        self.net = SegmentationUNet3D(1, 1, base, int(model_cfg.get("levels", 4)))

    def forward(self, x):
        if isinstance(x, dict):
            v0 = x["v0"]
            post = x["post"] if "post" in x else v0.repeat(1, self.t, 1, 1, 1)
        else:
            v0 = x[:, :1]
            post = x[:, 1 : self.t + 1] if x.shape[1] > 1 else v0.repeat(1, self.t, 1, 1, 1)
        if post.shape[1] < self.t:
            post = torch.cat([post, post[:, -1:].repeat(1, self.t - post.shape[1], 1, 1, 1)], dim=1)
        residual = post[:, : self.t] - v0
        fused = self.mix(torch.cat([self.anat(v0), self.kin(residual)], dim=1))
        return {"logits": self.net(fused)}
