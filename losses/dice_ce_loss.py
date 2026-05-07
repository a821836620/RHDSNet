from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary soft Dice loss for logits/targets shaped [B, 1, D, H, W]."""
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.ndim))
    intersection = torch.sum(prob * target, dim=dims)
    denom = torch.sum(prob + target, dim=dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class DiceCELoss(nn.Module):
    """L_seg = L_Dice + eta * BCEWithLogits."""

    def __init__(self, eta: float = 1.0):
        super().__init__()
        self.eta = float(eta)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        dice = soft_dice_loss(logits, target)
        ce = F.binary_cross_entropy_with_logits(logits, target)
        loss = dice + self.eta * ce
        return loss, {"dice_loss": float(dice.detach()), "ce_loss": float(ce.detach())}
