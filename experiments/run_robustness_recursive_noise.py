from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from datasets.dce_mri_dataset import DCEMRIDataset
from losses.metrics import logits_to_mask, patient_segmentation_metrics, summarize_metric_rows
from models.rhdsnet import RHDSNet
from utils.checkpoint import load_checkpoint
from utils.io import ensure_dir, save_csv, save_json
from utils.table_writer import write_table
from .common import get_device, move_batch


@torch.no_grad()
def _evaluate_setting(config, model, loader, device, phase, sigma, output_dir):
    rows = []
    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    model.eval()
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch["v0"], noise_perturb_phase=phase, noise_sigma=sigma)
        pred = logits_to_mask(out["logits"], threshold)[0, 0]
        gt = batch["mask"].detach().cpu().numpy()[0, 0]
        spacing = batch["spacing"].detach().cpu().numpy()[0]
        row = patient_segmentation_metrics(pred, gt, spacing)
        row["case_id"] = batch["case_id"][0] if isinstance(batch["case_id"], list) else batch["case_id"]
        rows.append(row)
    summary = summarize_metric_rows(rows, ["DSC", "HD95", "Sen"])
    save_csv(rows, Path(output_dir) / "patient_metrics.csv")
    save_json({"patients": rows, "summary": summary}, Path(output_dir) / "metrics.json")
    return summary


def run_robustness_recursive_noise(config: Dict, ckpt: str = None):
    output_dir = ensure_dir(config.get("experiment", {}).get("output_dir", "outputs/robustness_recursive_noise"))
    settings = config.get("experiment", {}).get("settings", [])
    sigma = float(config.get("experiment", {}).get("noise_sigma", 0.02))
    seed = config.get("training", {}).get("seeds", [2024])[0]
    device = get_device(config)
    loader = DataLoader(DCEMRIDataset(config, "test", seed=seed), batch_size=1, shuffle=False, num_workers=0)
    model = RHDSNet(config).to(device)
    ckpt = ckpt or str(Path(config.get("training", {}).get("output_dir", "outputs/rhdsnet")) / f"seed_{seed}" / "best.pth")
    if Path(ckpt).exists():
        load_checkpoint(ckpt, model, map_location=device)
    rows: List[Dict] = []
    for setting in settings:
        setting_dir = ensure_dir(Path(output_dir) / setting["name"].replace(" ", "_"))
        summary = _evaluate_setting(config, model, loader, device, setting.get("phase"), sigma, setting_dir)
        rows.append(
            {
                "Setting": setting["name"],
                "DSC": summary.get("DSC_mean", 0.0) * 100,
                "HD95": summary.get("HD95_mean", 0.0),
                "Sen": summary.get("Sen_mean", 0.0) * 100,
            }
        )
    write_table(rows, output_dir, "robustness_recursive_noise", ["Setting", "DSC", "HD95", "Sen"])
    return rows
