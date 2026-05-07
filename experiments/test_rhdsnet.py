from typing import Dict, Optional

import numpy as np

from engine.tester import Tester
from utils.io import save_json


def run_test(config: Dict, ckpt: Optional[str] = None):
    seeds = config.get("training", {}).get("seeds", [2024])
    results = []
    for seed in seeds:
        seed_ckpt = ckpt or f"{config.get('training', {}).get('output_dir', 'outputs/rhdsnet')}/seed_{seed}/best.pth"
        tester = Tester(config, ckpt=seed_ckpt, seed=seed, split="test")
        out = tester.run()
        results.append({"seed": seed, "summary": out["summary"], "output_dir": out["output_dir"]})
    aggregate = {}
    for key in ["DSC_mean", "HD95_mean", "Sen_mean", "mean-SSIM_mean"]:
        vals = np.asarray([r["summary"].get(key) for r in results if r["summary"].get(key) is not None], dtype=float)
        if vals.size:
            aggregate[key.replace("_mean", "_across_seed_mean")] = float(vals.mean())
            aggregate[key.replace("_mean", "_across_seed_std")] = float(vals.std(ddof=1 if vals.size > 1 else 0))
    save_json({"runs": results, "aggregate": aggregate}, f"{config.get('training', {}).get('output_dir', 'outputs/rhdsnet')}/five_seed_test_summary.json")
    return results
