from typing import Dict, Optional

import torch
import torch.nn as nn

from .mtc import MultiScaleTaskAlignedConditioning
from .rhdd import RHDD
from .segmentation_unet3d import SegmentationUNet3D


class RHDSNet(nn.Module):
    """RHDSNet = RHDD dynamics encoder + V0 segmentation U-Net + MTC fusion.

    During inference, forward(v0) uses only the single pre-contrast MRI V0.
    Real post-contrast phases are accepted only during training to compute the
    diffusion distillation loss.
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        model_cfg = config.get("model", {})
        train_cfg = config.get("training", {})
        rhdd_cfg = model_cfg.get("rhdd", {})
        mtc_cfg = model_cfg.get("mtc", {})
        self.levels = int(model_cfg.get("levels", 4))
        self.base_channels = int(model_cfg.get("base_channels", 24))
        self.channels = [self.base_channels * (2**i) for i in range(self.levels)]
        self.ddim_steps_train = int(rhdd_cfg.get("ddim_steps_train", 8))
        self.ddim_steps_test = int(rhdd_cfg.get("ddim_steps_test", 25))
        self.detach_hemo_features = bool(train_cfg.get("detach_hemo_features", False))

        self.rhdd = RHDD(config)
        self.segmentation = SegmentationUNet3D(
            in_channels=int(model_cfg.get("in_channels", 1)),
            out_channels=int(model_cfg.get("num_classes", 1)),
            base_channels=self.base_channels,
            levels=self.levels,
        )
        self.mtc = MultiScaleTaskAlignedConditioning(
            channels=self.channels,
            inject_layers=mtc_cfg.get("inject_layers", [1, 2, 3, 4]),
            attn_channels=int(mtc_cfg.get("attn_channels", 16)),
            kv_pool_size=tuple(mtc_cfg.get("kv_pool_size", [2, 4, 4])),
            enabled=bool(mtc_cfg.get("enabled", True)),
            chunk_size=int(mtc_cfg.get("chunk_size", 32768)),
        )

    def forward(
        self,
        v0: torch.Tensor,
        post: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        epoch: int = 0,
        stage: str = "joint",
        sample_steps: Optional[int] = None,
        noise_perturb_phase: Optional[int] = None,
        noise_sigma: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        outputs = {}
        if post is not None and mask is not None and stage in ("pretrain", "joint"):
            diff_loss, diff_logs = self.rhdd.training_losses(v0, post, mask, epoch=epoch)
            outputs["diff_loss"] = diff_loss
            outputs["diff_logs"] = diff_logs
            if stage == "pretrain":
                return outputs

        steps = sample_steps if sample_steps is not None else (self.ddim_steps_train if self.training else self.ddim_steps_test)
        hemo_features, generated = self.rhdd.encode_dynamics(
            v0,
            ddim_steps=steps,
            noise_perturb_phase=noise_perturb_phase,
            noise_sigma=noise_sigma,
        )
        if self.detach_hemo_features:
            hemo_features = [[feat.detach() for feat in phase_feats] for phase_feats in hemo_features]
            generated = generated.detach()
        spatial_features = self.segmentation.encode(v0)
        fused_features = self.mtc(spatial_features, hemo_features)
        outputs["logits"] = self.segmentation.decode(fused_features)
        outputs["generated"] = generated
        return outputs
