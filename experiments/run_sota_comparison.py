from copy import deepcopy
from pathlib import Path
from typing import Dict, List

from models.baselines import get_baseline_model
from models.rhdsnet import RHDSNet
from utils.checkpoint import load_checkpoint
from utils.io import ensure_dir
from utils.table_writer import write_table
from engine.trainer import Trainer
from .common import (
    build_loaders,
    evaluate_segmentation_model,
    get_device,
    model_complexity_row,
    summary_to_percent_fields,
    train_segmentation_proxy,
)


METHODS = [
    ("nnUNet-like baseline", "3D MIP/V0"),
    ("MedSegDiff-V2-like baseline", "3D MIP/V0"),
    ("Tumor-sen-like two-phase model", "V0+V1"),
    ("ALMN-like two-phase model", "V0+V1"),
    ("TSESNet-like 4D temporal fusion model", "V0...VT"),
    ("DEFNet-like dual encoder model", "V0...VT"),
    ("PLHN-like prototype/hybrid temporal model", "V0...VT"),
    ("AKF-Net-like anatomical-kinetic fusion model", "V0...VT"),
    ("BBDiffSF-like diffusion semantic fusion model", "V0...VT"),
    ("RHDSNet", "V0 only"),
]


def run_sota_comparison(config: Dict):
    """Train/evaluate SOTA proxy baselines and RHDSNet on the configured dataset."""
    output_dir = ensure_dir(config.get("experiment", {}).get("output_dir", Path(config.get("training", {}).get("output_dir", "outputs")) / "sota"))
    dataset_name = config.get("dataset", {}).get("name", "ISPY2").upper()
    device = get_device(config)
    seed = config.get("training", {}).get("seeds", [2024])[0]
    _, _, test_loader = build_loaders(config, seed)
    rows: List[Dict] = []
    for method, input_desc in METHODS:
        run_cfg = deepcopy(config)
        method_dir = ensure_dir(Path(output_dir) / method.replace("/", "_").replace(" ", "_"))
        if method == "RHDSNet":
            model = RHDSNet(run_cfg).to(device)
            ckpt = Path(run_cfg.get("training", {}).get("output_dir", "outputs/rhdsnet")) / f"seed_{seed}" / "best.pth"
            if not ckpt.exists():
                Trainer(run_cfg, seed=seed).train()
            if ckpt.exists():
                load_checkpoint(ckpt, model, map_location=device)
            result = evaluate_segmentation_model(run_cfg, model, test_loader, device, save_dir=method_dir)
        else:
            model = get_baseline_model(method, run_cfg)
            model = train_segmentation_proxy(run_cfg, model, method_dir, seed=seed)
            result = evaluate_segmentation_model(run_cfg, model, test_loader, device, save_dir=method_dir)
        row = {"Methods": method, "Input": input_desc}
        row.update({"ISPY2 DSC": "", "ISPY2 HD95": "", "ISPY2 Sen": "", "DUKE DSC": "", "DUKE HD95": "", "DUKE Sen": ""})
        prefix = "DUKE" if dataset_name == "DUKE" else "ISPY2"
        row.update(summary_to_percent_fields(result["summary"], prefix))
        row.update(model_complexity_row(model, config))
        rows.append(row)
    fields = ["Methods", "Input", "ISPY2 DSC", "ISPY2 HD95", "ISPY2 Sen", "DUKE DSC", "DUKE HD95", "DUKE Sen", "Params", "FLOPs"]
    write_table(rows, output_dir, "sota_comparison", fields)
    return rows
