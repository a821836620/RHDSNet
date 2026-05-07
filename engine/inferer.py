from typing import Dict, Optional

import torch

from models.rhdsnet import RHDSNet
from utils.checkpoint import load_checkpoint


class Inferer:
    """Small wrapper for V0-only inference from tensors."""

    def __init__(self, config: Dict, ckpt: Optional[str] = None, device: str = "cuda"):
        self.device = torch.device("cuda" if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = RHDSNet(config).to(self.device)
        if ckpt:
            load_checkpoint(ckpt, self.model, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, v0: torch.Tensor, threshold: float = 0.5):
        v0 = v0.to(self.device)
        out = self.model(v0)
        prob = torch.sigmoid(out["logits"])
        return {"prob": prob, "mask": (prob >= threshold).float(), "generated": out["generated"]}
