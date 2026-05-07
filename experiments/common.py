from pathlib import Path
from typing import Dict, Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from datasets.dce_mri_dataset import DCEMRIDataset
from losses.dice_ce_loss import DiceCELoss
from losses.metrics import logits_to_mask, mean_ssim, patient_segmentation_metrics, summarize_metric_rows
from utils.checkpoint import save_checkpoint
from utils.flops_params import estimate_flops
from utils.io import ensure_dir, save_csv, save_json
from utils.seed import seed_everything


def get_device(config):
    requested = config.get("training", {}).get("device", "cuda")
    return torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")


def build_loaders(config: Dict, seed: int):
    train_ds = DCEMRIDataset(config, "train", seed=seed)
    val_ds = DCEMRIDataset(config, "val", seed=seed)
    test_ds = DCEMRIDataset(config, "test", seed=seed)
    bs = int(config.get("training", {}).get("batch_size", 1))
    nw = int(config.get("training", {}).get("num_workers", 2))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


def move_batch(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def forward_for_experiment(model, batch):
    """Dispatch RHDSNet with V0-only tensors and proxy baselines with dict batches."""
    if hasattr(model, "rhdd") and hasattr(model, "segmentation"):
        return model(batch["v0"])
    return model(batch)


def train_segmentation_proxy(config: Dict, model, output_dir, seed: int = 2024, max_epochs: Optional[int] = None):
    """Train a baseline/proxy segmentation model with Dice+CE."""
    seed_everything(seed)
    device = get_device(config)
    output_dir = ensure_dir(output_dir)
    train_loader, val_loader, _ = build_loaders(config, seed)
    model = model.to(device)
    criterion = DiceCELoss(float(config.get("loss", {}).get("eta", 1.0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.get("training", {}).get("lr", 1e-4)))
    scaler = GradScaler(enabled=bool(config.get("training", {}).get("amp", True)) and device.type == "cuda")
    max_epochs = int(max_epochs or config.get("experiment", {}).get("proxy_epochs", config.get("training", {}).get("max_epochs", 1000)))
    patience = int(config.get("training", {}).get("early_stopping_patience", 50))
    best = -1.0
    wait = 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                out = forward_for_experiment(model, batch)
                loss, _ = criterion(out["logits"], batch["mask"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
            count += 1
        val = evaluate_segmentation_model(config, model, val_loader, device, save_dir=None)
        dsc = val["summary"].get("DSC_mean", 0.0)
        history.append({"epoch": epoch, "loss": total / max(count, 1), **val["summary"]})
        if dsc > best:
            best = dsc
            wait = 0
            save_checkpoint(output_dir / "best.pth", model, optimizer=optimizer, epoch=epoch, best_metric=best, config=config)
        else:
            wait += 1
        if wait >= patience:
            break
    save_csv(history, output_dir / "train_log.csv")
    save_json(history, output_dir / "train_log.json")
    return model


@torch.no_grad()
def evaluate_segmentation_model(config: Dict, model, loader, device, save_dir=None, generated_from_output: bool = False):
    model.eval()
    rows = []
    threshold = float(config.get("inference", {}).get("threshold", 0.5))
    if save_dir:
        save_dir = ensure_dir(save_dir)
    for batch in loader:
        batch = move_batch(batch, device)
        out = forward_for_experiment(model, batch)
        pred = logits_to_mask(out["logits"], threshold)[0, 0]
        gt = batch["mask"].detach().cpu().numpy()[0, 0]
        spacing = batch["spacing"].detach().cpu().numpy()[0]
        row = patient_segmentation_metrics(pred, gt, spacing)
        case_id = batch["case_id"][0] if isinstance(batch["case_id"], list) else batch["case_id"]
        row["case_id"] = case_id
        if generated_from_output and "generated" in out:
            row["mean-SSIM"] = mean_ssim(batch["post"].detach().cpu().numpy()[0], out["generated"].detach().cpu().numpy()[0])
        rows.append(row)
    summary = summarize_metric_rows(rows, ["DSC", "HD95", "Sen", "mean-SSIM"])
    if save_dir:
        save_csv(rows, save_dir / "patient_metrics.csv")
        save_json({"patients": rows, "summary": summary}, save_dir / "metrics.json")
    return {"patients": rows, "summary": summary}


def summary_to_percent_fields(summary: Dict, prefix: str):
    return {
        f"{prefix} DSC": summary.get("DSC_mean", float("nan")) * 100.0,
        f"{prefix} HD95": summary.get("HD95_mean", float("nan")),
        f"{prefix} Sen": summary.get("Sen_mean", float("nan")) * 100.0,
    }


def model_complexity_row(model, config):
    roi = config.get("inference", {}).get("roi_size", [96, 96, 48])
    shape = (1, 1, int(roi[2]), int(roi[0]), int(roi[1]))
    flops, params = estimate_flops(model, input_shape=shape, device="cpu")
    return {"Params": params, "FLOPs": flops}
