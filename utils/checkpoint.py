from pathlib import Path
from typing import Any, Dict, Optional

import torch

from .io import ensure_dir


def save_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    epoch: int = 0,
    best_metric: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a checkpoint that can be resumed or used for inference."""
    ensure_dir(Path(path).parent)
    state = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        "config": config,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    torch.save(state, path)


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, map_location="cpu") -> Dict[str, Any]:
    """Load a checkpoint and optionally restore model/optimizer/scheduler states."""
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(ckpt["model"], strict=False)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
