from typing import Dict

import numpy as np

from engine.trainer import Trainer
from utils.io import save_json


def run_train(config: Dict):
    """Run five independent RHDSNet training seeds by default."""
    seeds = config.get("training", {}).get("seeds", [2024])
    results = []
    for seed in seeds:
        trainer = Trainer(config, seed=seed)
        results.append({"seed": seed, **trainer.train()})
    best = np.asarray([r["best_dice"] for r in results], dtype=float)
    aggregate = {
        "best_dice_mean": float(best.mean()) if best.size else 0.0,
        "best_dice_std": float(best.std(ddof=1 if best.size > 1 else 0)) if best.size else 0.0,
    }
    save_json({"runs": results, "aggregate": aggregate}, f"{config.get('training', {}).get('output_dir', 'outputs/rhdsnet')}/five_seed_train_summary.json")
    return results
