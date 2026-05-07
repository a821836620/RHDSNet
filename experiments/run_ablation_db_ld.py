from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np

from engine.tester import Tester
from engine.trainer import Trainer
from utils.io import ensure_dir
from utils.table_writer import write_table


def run_ablation_db_ld(config: Dict):
    output_dir = ensure_dir(config.get("experiment", {}).get("output_dir", "outputs/ablation_db_ld"))
    settings = config.get("experiment", {}).get("settings", [])
    seeds = config.get("experiment", {}).get("seeds", config.get("training", {}).get("seeds", [2024]))
    rows: List[Dict] = []
    seed_rows: List[Dict] = []
    for setting in settings:
        metric_acc = []
        for seed in seeds:
            run_cfg = deepcopy(config)
            run_cfg["training"]["seeds"] = [seed]
            run_cfg["training"]["output_dir"] = str(Path(output_dir) / setting["name"].replace(" ", "_").replace("+", "plus"))
            run_cfg["model"]["rhdd"]["dynamic_blending"] = bool(setting["dynamic_blending"])
            run_cfg["model"]["rhdd"]["local_decoupling"] = bool(setting["local_decoupling"])
            trainer = Trainer(run_cfg, seed=seed)
            train_out = trainer.train()
            tester = Tester(run_cfg, ckpt=str(Path(train_out["output_dir"]) / "best.pth"), seed=seed)
            result = tester.run()
            summary = result["summary"]
            one = {
                "Method": setting["name"],
                "seed": seed,
                "mean-SSIM": summary.get("mean-SSIM_mean", 0.0),
                "DSC": summary.get("DSC_mean", 0.0) * 100,
                "HD95": summary.get("HD95_mean", 0.0),
                "Sen": summary.get("Sen_mean", 0.0) * 100,
            }
            seed_rows.append(one)
            metric_acc.append(one)
        rows.append(
            {
                "Method": setting["name"],
                "mean-SSIM": float(np.mean([m["mean-SSIM"] for m in metric_acc])),
                "DSC": float(np.mean([m["DSC"] for m in metric_acc])),
                "HD95": float(np.mean([m["HD95"] for m in metric_acc])),
                "Sen": float(np.mean([m["Sen"] for m in metric_acc])),
            }
        )
    write_table(rows, output_dir, "ablation_db_ld", ["Method", "mean-SSIM", "DSC", "HD95", "Sen"])
    write_table(seed_rows, output_dir, "ablation_db_ld_per_seed", ["Method", "seed", "mean-SSIM", "DSC", "HD95", "Sen"])
    return rows
