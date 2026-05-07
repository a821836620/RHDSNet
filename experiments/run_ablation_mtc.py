from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import numpy as np

from engine.tester import Tester
from engine.trainer import Trainer
from utils.io import ensure_dir
from utils.table_writer import write_table


def run_ablation_mtc(config: Dict):
    output_dir = ensure_dir(config.get("experiment", {}).get("output_dir", "outputs/ablation_mtc"))
    layer_sets = config.get("experiment", {}).get("layer_sets", [[1, 2, 3, 4]])
    seeds = config.get("experiment", {}).get("seeds", config.get("training", {}).get("seeds", [2024]))
    rows: List[Dict] = []
    seed_rows: List[Dict] = []
    for layers in layer_sets:
        tag = "hf" + "_".join(str(x) for x in layers)
        metric_acc = []
        for seed in seeds:
            run_cfg = deepcopy(config)
            run_cfg["training"]["seeds"] = [seed]
            run_cfg["training"]["output_dir"] = str(Path(output_dir) / tag)
            run_cfg["model"]["mtc"]["inject_layers"] = layers
            trainer = Trainer(run_cfg, seed=seed)
            train_out = trainer.train()
            tester = Tester(run_cfg, ckpt=str(Path(train_out["output_dir"]) / "best.pth"), seed=seed)
            result = tester.run()
            summary = result["summary"]
            one = {
                "hf1": int(1 in layers),
                "hf2": int(2 in layers),
                "hf3": int(3 in layers),
                "hf4": int(4 in layers),
                "seed": seed,
                "DSC": summary.get("DSC_mean", 0.0) * 100,
                "HD95": summary.get("HD95_mean", 0.0),
                "Sen": summary.get("Sen_mean", 0.0) * 100,
            }
            seed_rows.append(one)
            metric_acc.append(one)
        rows.append(
            {
                "hf1": int(1 in layers),
                "hf2": int(2 in layers),
                "hf3": int(3 in layers),
                "hf4": int(4 in layers),
                "DSC": float(np.mean([m["DSC"] for m in metric_acc])),
                "HD95": float(np.mean([m["HD95"] for m in metric_acc])),
                "Sen": float(np.mean([m["Sen"] for m in metric_acc])),
            }
        )
    write_table(rows, output_dir, "ablation_mtc", ["hf1", "hf2", "hf3", "hf4", "DSC", "HD95", "Sen"])
    write_table(seed_rows, output_dir, "ablation_mtc_per_seed", ["hf1", "hf2", "hf3", "hf4", "seed", "DSC", "HD95", "Sen"])
    return rows
