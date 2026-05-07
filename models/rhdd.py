from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.diffusion_loss import local_decoupling_gate, spatial_weight_from_gate, weighted_noise_mse
from .diffusion_unet3d import DiffusionUNet3D


def _extract(buffer: torch.Tensor, t: torch.Tensor, x_shape) -> torch.Tensor:
    """Gather per-step scalars and reshape for broadcasting to x_shape."""
    out = buffer.gather(0, t.long())
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class RHDD(nn.Module):
    """Residual Hemodynamic Dynamics Distillation.

    This recurrent conditional diffusion module learns V1...VT from V0 by
    conditioning phase i on the previous generated/blended context volume.
    """

    def __init__(self, config: Dict):
        super().__init__()
        rhdd_cfg = config.get("model", {}).get("rhdd", {})
        model_cfg = config.get("model", {})
        self.num_post_phases = int(model_cfg.get("num_post_phases", rhdd_cfg.get("num_post_phases", 4)))
        self.num_steps = int(rhdd_cfg.get("diffusion_steps", 1000))
        self.dynamic_blending = bool(rhdd_cfg.get("dynamic_blending", True))
        self.blending_delta = float(rhdd_cfg.get("blending_delta", 1.0))
        self.total_blending_epochs = int(rhdd_cfg.get("total_blending_epochs", config.get("training", {}).get("max_epochs", 1000)))
        self.local_decoupling = bool(rhdd_cfg.get("local_decoupling", True))
        self.kappa = float(rhdd_cfg.get("kappa", 12.0))
        self.tau = float(rhdd_cfg.get("tau", 0.35))
        self.lambda_bg = float(rhdd_cfg.get("lambda_bg", 0.2))
        base_channels = int(model_cfg.get("base_channels", 24))
        levels = int(model_cfg.get("levels", 4))

        self.denoiser = DiffusionUNet3D(
            in_channels=2,
            out_channels=1,
            base_channels=base_channels,
            levels=levels,
        )
        betas = torch.linspace(
            float(rhdd_cfg.get("beta_start", 1e-6)),
            float(rhdd_cfg.get("beta_end", 1e-2)),
            self.num_steps,
            dtype=torch.float32,
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """z_s = sqrt(alpha_bar_s) * x0 + sqrt(1-alpha_bar_s) * epsilon."""
        return _extract(self.sqrt_alpha_bars, t, x0.shape) * x0 + _extract(
            self.sqrt_one_minus_alpha_bars, t, x0.shape
        ) * noise

    def predict_x0(self, z_s: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        return (z_s - _extract(self.sqrt_one_minus_alpha_bars, t, z_s.shape) * eps_pred) / (
            _extract(self.sqrt_alpha_bars, t, z_s.shape) + 1e-8
        )

    def blending_beta(self, epoch: int) -> float:
        """beta = max(0, delta * (1 - epoch / total))."""
        if not self.dynamic_blending:
            return 0.0
        return max(0.0, self.blending_delta * (1.0 - float(epoch) / max(float(self.total_blending_epochs), 1.0)))

    def training_losses(self, v0: torch.Tensor, post: torch.Tensor, mask: torch.Tensor, epoch: int = 0):
        """Compute recurrent LD diffusion loss over all post-contrast phases.

        post shape is [B, T, D, H, W]. Each target phase is viewed as [B,1,D,H,W].
        """
        b = v0.shape[0]
        prev_context = v0
        losses = []
        logs = {}
        beta_blend = self.blending_beta(epoch)
        for phase_idx in range(self.num_post_phases):
            target = post[:, phase_idx : phase_idx + 1]
            t = torch.randint(0, self.num_steps, (b,), device=v0.device, dtype=torch.long)
            noise = torch.randn_like(target)
            z_s = self.q_sample(target, t, noise)
            gate = None
            weight = None
            if self.local_decoupling:
                gate = local_decoupling_gate(target, v0, mask, self.kappa, self.tau)
                weight = spatial_weight_from_gate(gate, self.lambda_bg)
            eps_pred, _ = self.denoiser(z_s, t, prev_context, feature_gate=gate, return_features=True)
            loss_i = weighted_noise_mse(eps_pred, noise, weight)
            losses.append(loss_i)
            logs[f"diff_phase_{phase_idx + 1}"] = float(loss_i.detach())

            # One-step x0 estimate is used as the generated context for the next phase.
            v_hat = self.predict_x0(z_s, t, eps_pred).detach()
            if self.dynamic_blending:
                prev_context = (1.0 - beta_blend) * v_hat + beta_blend * target
            else:
                prev_context = v_hat
        loss = torch.stack(losses).mean()
        logs["diff_loss"] = float(loss.detach())
        logs["dynamic_blending_beta"] = beta_blend
        return loss, logs

    def _features_at_x(self, x: torch.Tensor, prev_context: torch.Tensor, v0: torch.Tensor) -> List[torch.Tensor]:
        t0 = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        gate = None
        if self.local_decoupling:
            # In inference real Vi/Y are unavailable, so gate is estimated from |V_hat_i - V0|.
            gate = local_decoupling_gate(x, v0, torch.ones_like(v0), self.kappa, self.tau)
        _, features = self.denoiser(x, t0, prev_context, feature_gate=gate, return_features=True)
        return features

    def sample_next_phase(self, prev_context: torch.Tensor, v0: torch.Tensor, ddim_steps: int = 25):
        """DDIM sampling for one post-contrast phase conditioned on previous context."""
        steps = int(max(1, min(ddim_steps, self.num_steps)))
        times = torch.linspace(self.num_steps - 1, 0, steps, device=prev_context.device).long()
        x = torch.randn_like(prev_context)
        for idx, step in enumerate(times):
            t = torch.full((prev_context.shape[0],), int(step.item()), device=prev_context.device, dtype=torch.long)
            eps_pred, _ = self.denoiser(x, t, prev_context, return_features=False)
            a_t = _extract(self.alpha_bars, t, x.shape)
            x0_pred = (x - torch.sqrt(1.0 - a_t) * eps_pred) / (torch.sqrt(a_t) + 1e-8)
            if idx == len(times) - 1:
                a_prev = torch.ones_like(a_t)
            else:
                prev_t = torch.full(
                    (prev_context.shape[0],),
                    int(times[idx + 1].item()),
                    device=prev_context.device,
                    dtype=torch.long,
                )
                a_prev = _extract(self.alpha_bars, prev_t, x.shape)
            x = torch.sqrt(a_prev) * x0_pred + torch.sqrt(torch.clamp(1.0 - a_prev, min=0.0)) * eps_pred
        features = self._features_at_x(x, prev_context, v0)
        return x, features

    def encode_dynamics(
        self,
        v0: torch.Tensor,
        ddim_steps: int = 25,
        noise_perturb_phase: Optional[int] = None,
        noise_sigma: float = 0.02,
    ) -> Tuple[List[List[torch.Tensor]], torch.Tensor]:
        """Generate V1...VT recursively and return RHDD encoder features.

        Inference uses only V0. If noise_perturb_phase is set (1-based), Gaussian
        noise is injected after that phase and recursion continues from the noisy phase.
        """
        prev_context = v0
        hemo_features: List[List[torch.Tensor]] = []
        generated = []
        for phase_idx in range(1, self.num_post_phases + 1):
            v_hat, features = self.sample_next_phase(prev_context, v0, ddim_steps=ddim_steps)
            if noise_perturb_phase is not None and phase_idx == int(noise_perturb_phase):
                v_hat = v_hat + float(noise_sigma) * torch.randn_like(v_hat)
                features = self._features_at_x(v_hat, prev_context, v0)
            hemo_features.append(features)
            generated.append(v_hat)
            prev_context = v_hat
        return hemo_features, torch.cat(generated, dim=1)  # [B, T, D, H, W]
