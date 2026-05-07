from .fourd_unet import FourDUNetProxy, TemporalFusionProxy
from .mip_unet import MIPUNetProxy
from .nnunet_like import MedSegDiffV2Proxy, NNUNetLikeProxy
from .two_phase_unet import ALMNProxy, TwoPhaseUNetProxy


def get_baseline_model(name: str, config):
    """Factory for reproducible proxy baselines."""
    key = name.lower()
    if "nnunet" in key:
        return NNUNetLikeProxy(config)
    if "medsegdiff" in key:
        return MedSegDiffV2Proxy(config)
    if "mip" in key:
        return MIPUNetProxy(config)
    if "tumor" in key or "two-phase" in key:
        return TwoPhaseUNetProxy(config)
    if "almn" in key:
        return ALMNProxy(config)
    if any(x in key for x in ["tsesnet", "4d", "plhn"]):
        return FourDUNetProxy(config)
    if any(x in key for x in ["defnet", "akf", "bbdiffsf"]):
        return TemporalFusionProxy(config)
    raise ValueError(f"Unknown baseline: {name}")
