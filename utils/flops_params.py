from typing import Tuple

import torch


def count_params(model) -> float:
    """Return trainable parameter count in millions."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def estimate_flops(model, input_shape=(1, 1, 48, 96, 96), device="cpu") -> Tuple[float, float]:
    """Return FLOPs(G) and Params(M). Uses thop if installed, otherwise returns Params only."""
    params_m = count_params(model)
    try:
        from thop import profile

        model = model.to(device)
        dummy = torch.randn(*input_shape, device=device)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops / 1e9, params / 1e6
    except Exception:
        return 0.0, params_m
