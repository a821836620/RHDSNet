import torch
import torch.nn.functional as F


def per_volume_minmax(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample min-max normalization over D,H,W for tensors [B, 1, D, H, W]."""
    dims = tuple(range(2, x.ndim))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)


def local_decoupling_gate(v_i, v0, mask, kappa: float, tau: float) -> torch.Tensor:
    """M_i = sigmoid(kappa * (Norm(|V_i - V0|) * Y - tau))."""
    residual = torch.abs(v_i - v0)
    norm_residual = per_volume_minmax(residual)
    return torch.sigmoid(float(kappa) * (norm_residual * mask - float(tau)))


def spatial_weight_from_gate(gate: torch.Tensor, lambda_bg: float) -> torch.Tensor:
    """W_i = lambda + (1 - lambda) * M_i."""
    lam = float(lambda_bg)
    return lam + (1.0 - lam) * gate


def weighted_noise_mse(eps_pred: torch.Tensor, eps: torch.Tensor, weight: torch.Tensor = None) -> torch.Tensor:
    """LD diffusion objective ||sqrt(W) * (eps - eps_pred)||_2^2."""
    if weight is None:
        return F.mse_loss(eps_pred, eps)
    return torch.mean(weight * (eps_pred - eps) ** 2)
