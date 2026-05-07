import torch
import torch.nn as nn

from models.segmentation_unet3d import SegmentationUNet3D


class NNUNetLikeProxy(nn.Module):
    """nnUNet-like 3D U-Net proxy for V0-only segmentation.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        self.net = SegmentationUNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=int(model_cfg.get("base_channels", 24)),
            levels=int(model_cfg.get("levels", 4)),
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x["v0"]
        if x.ndim == 4:
            x = x[:, None]
        if x.shape[1] != 1:
            x = x[:, :1]
        return {"logits": self.net(x)}


class MedSegDiffV2Proxy(nn.Module):
    """Simplified diffusion-segmentation proxy.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation. The denoising branch is approximated by a
    residual refinement head on top of a 3D U-Net segmentation backbone.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        base = int(model_cfg.get("base_channels", 24))
        self.backbone = SegmentationUNet3D(1, 1, base, int(model_cfg.get("levels", 4)))
        self.refine = nn.Sequential(
            nn.Conv3d(2, base, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(base, 1, 1),
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x["v0"]
        if x.shape[1] != 1:
            x = x[:, :1]
        coarse = self.backbone(x)
        return {"logits": coarse + self.refine(torch.cat([x, coarse], dim=1))}
