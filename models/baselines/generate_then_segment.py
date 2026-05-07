import torch
import torch.nn as nn

from .fourd_unet import FourDUNetProxy
from .simple_diffusion_generator import SimplePhaseGenerator


class NamedGeneratorProxy(SimplePhaseGenerator):
    """TeNCA/DCEtriFormer/SPHERE-like generator proxy.

    This is a faithful proxy implementation for experimental comparison, not
    the official implementation. The class name records the experimental row;
    architecture remains controlled and reproducible for fair comparison.
    """

    def __init__(self, config, name: str):
        model_cfg = config.get("model", {})
        super().__init__(int(model_cfg.get("num_post_phases", 4)), int(model_cfg.get("base_channels", 24)))
        self.name = name


class GenerateThenSegment(nn.Module):
    """Generate V1...VT from V0 and feed a fixed 4D segmentation model."""

    def __init__(self, config, generator_name: str = "DDPM generator baseline"):
        super().__init__()
        self.generator = NamedGeneratorProxy(config, generator_name)
        self.segmenter = FourDUNetProxy(config)

    def forward(self, x):
        if isinstance(x, dict):
            v0 = x["v0"]
        else:
            v0 = x[:, :1]
        post_hat = self.generator(v0)
        logits = self.segmenter({"v0": v0, "post": post_hat})["logits"]
        return {"logits": logits, "generated": post_hat}
