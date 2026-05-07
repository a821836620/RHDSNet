import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets.dce_mri_dataset import DCEMRIDataset
from losses.dice_ce_loss import DiceCELoss
from losses.metrics import logits_to_mask, patient_segmentation_metrics, summarize_metric_rows
from models.rhdsnet import RHDSNet
from utils.checkpoint import save_checkpoint
from utils.io import ensure_dir, save_config, save_csv, save_json
from utils.logger import get_logger
from utils.seed import seed_everything


def move_batch_to_device(batch: Dict, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out


class Trainer:
    """End-to-end RHDSNet trainer with diffusion pretraining and joint training."""

    def __init__(self, config: Dict, seed: int = 2024):
        self.config = config
        self.seed = int(seed)
        seed_everything(self.seed)
        train_cfg = config.get("training", {})
        requested = train_cfg.get("device", "cuda")
        self.distributed = bool(train_cfg.get("distributed", False)) and int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if self.distributed:
            torch.cuda.set_device(self.local_rank)
            torch.distributed.init_process_group(backend="nccl")
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
        self.is_main_process = self.rank == 0
        self.output_dir = ensure_dir(Path(train_cfg.get("output_dir", "outputs/rhdsnet")) / f"seed_{self.seed}")
        self.logger = get_logger("RHDSNetTrainer", self.output_dir / "train.log")
        if self.is_main_process:
            save_config(config, self.output_dir / "config_snapshot.yaml")

        self.train_ds = DCEMRIDataset(config, split="train", seed=self.seed)
        self.val_ds = DCEMRIDataset(config, split="val", seed=self.seed)
        self.train_sampler = DistributedSampler(self.train_ds, shuffle=True, seed=self.seed) if self.distributed else None
        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=int(train_cfg.get("batch_size", 1)),
            shuffle=self.train_sampler is None,
            sampler=self.train_sampler,
            num_workers=int(train_cfg.get("num_workers", 2)),
            pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=0)

        self.model = RHDSNet(config).to(self.device)
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)
        elif train_cfg.get("data_parallel", False) and torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.criterion = DiceCELoss(eta=float(config.get("loss", {}).get("eta", 1.0)))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg.get("lr", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        self.scaler = GradScaler(enabled=bool(train_cfg.get("amp", True)) and self.device.type == "cuda")
        self.max_epochs = int(train_cfg.get("max_epochs", 1000))
        self.pretrain_epochs = int(train_cfg.get("diffusion_pretrain_epochs", 200))
        self.patience = int(train_cfg.get("early_stopping_patience", 50))
        self.gamma = float(config.get("loss", {}).get("gamma", 0.5))
        self.history = []

    @property
    def _model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def train(self):
        best_dice = -1.0
        no_improve = 0
        for epoch in range(self.max_epochs):
            start = time.time()
            train_log = self.train_one_epoch(epoch)
            if epoch < self.pretrain_epochs and not self.config.get("training", {}).get("validate_during_pretrain", False):
                val_summary = {"DSC_mean": 0.0, "HD95_mean": np.nan, "Sen_mean": 0.0}
            else:
                val_summary = self.validate(epoch)
            dice = val_summary.get("DSC_mean", 0.0)
            row = {"epoch": epoch, **train_log, **val_summary, "seconds": time.time() - start}
            self.history.append(row)
            if self.is_main_process:
                save_csv(self.history, self.output_dir / "train_log.csv")
                save_json(self.history, self.output_dir / "train_log.json")

            if epoch >= self.pretrain_epochs and dice > best_dice:
                best_dice = dice
                no_improve = 0
                if self.is_main_process:
                    save_checkpoint(self.output_dir / "best.pth", self.model, self.optimizer, epoch=epoch, best_metric=best_dice, config=self.config)
            else:
                no_improve += 1
            if self.is_main_process and epoch % int(self.config.get("training", {}).get("save_every", 50)) == 0:
                save_checkpoint(self.output_dir / f"epoch_{epoch:04d}.pth", self.model, self.optimizer, epoch=epoch, best_metric=best_dice, config=self.config)
            if self.is_main_process:
                self.logger.info(
                    f"epoch={epoch:04d} train_loss={train_log['loss']:.4f} val_dsc={dice * 100:.2f} best={best_dice * 100:.2f}"
                )
            if epoch >= self.pretrain_epochs and no_improve >= self.patience:
                if self.is_main_process:
                    self.logger.info(f"Early stopping at epoch {epoch}.")
                break
        if self.is_main_process:
            save_checkpoint(self.output_dir / "last.pth", self.model, self.optimizer, epoch=len(self.history) - 1, best_metric=best_dice, config=self.config)
        return {"best_dice": best_dice, "output_dir": str(self.output_dir)}

    def train_one_epoch(self, epoch: int):
        self.model.train()
        if hasattr(self.train_ds, "set_epoch"):
            self.train_ds.set_epoch(epoch)
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
        stage = "pretrain" if epoch < self.pretrain_epochs else "joint"
        sums = {"loss": 0.0, "seg_loss": 0.0, "diff_loss": 0.0}
        n = 0
        for batch in self.train_loader:
            batch = move_batch_to_device(batch, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(batch["v0"], batch["post"], batch["mask"], epoch=epoch, stage=stage)
                diff_loss = outputs.get("diff_loss", torch.tensor(0.0, device=self.device))
                if stage == "pretrain":
                    loss = diff_loss
                    seg_loss = torch.tensor(0.0, device=self.device)
                else:
                    seg_loss, _ = self.criterion(outputs["logits"], batch["mask"])
                    loss = self.gamma * diff_loss + (1.0 - self.gamma) * seg_loss
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            sums["loss"] += float(loss.detach())
            sums["seg_loss"] += float(seg_loss.detach())
            sums["diff_loss"] += float(diff_loss.detach())
            n += 1
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["stage"] = stage
        return out

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        rows = []
        for batch in self.val_loader:
            batch = move_batch_to_device(batch, self.device)
            outputs = self.model(batch["v0"], sample_steps=self._model.ddim_steps_train)
            pred = logits_to_mask(outputs["logits"], self.config.get("inference", {}).get("threshold", 0.5))[0, 0]
            gt = batch["mask"].detach().cpu().numpy()[0, 0]
            spacing = batch["spacing"].detach().cpu().numpy()[0]
            rows.append(patient_segmentation_metrics(pred, gt, spacing))
        summary = summarize_metric_rows(rows, ["DSC", "HD95", "Sen"])
        out = {f"val_{k}": v for k, v in summary.items()}
        out.update(
            {
                "DSC_mean": summary.get("DSC_mean", 0.0),
                "HD95_mean": summary.get("HD95_mean", np.nan),
                "Sen_mean": summary.get("Sen_mean", 0.0),
            }
        )
        return out
