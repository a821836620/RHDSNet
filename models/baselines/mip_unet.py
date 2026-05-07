import torch
import torch.nn as nn

from models.segmentation_unet3d import SegmentationUNet3D


class MIPUNetProxy(nn.Module):
    """3D MIP-based proxy baseline.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation. Multi-phase input is reduced by channel-wise
    maximum intensity projection and segmented by a 3D U-Net.
    """

    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        self.net = SegmentationUNet3D(1, 1, int(model_cfg.get("base_channels", 24)), int(model_cfg.get("levels", 4)))

    def forward(self, x):
        if isinstance(x, dict):
            x = torch.cat([x["v0"], x["post"]], dim=1) if "post" in x else x["v0"]
        if x.shape[1] > 1:
            x = torch.max(x, dim=1, keepdim=True).values
        return {"logits": self.net(x)}
