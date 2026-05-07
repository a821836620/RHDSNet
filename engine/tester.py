from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dce_mri_dataset import DCEMRIDataset
from losses.metrics import logits_to_mask, mean_ssim, patient_segmentation_metrics, summarize_metric_rows
from models.rhdsnet import RHDSNet
from utils.checkpoint import load_checkpoint
from utils.io import crop_size_to_dhw, ensure_dir, save_csv, save_json, save_volume
from utils.logger import get_logger
from .sliding_window import sliding_window_inference
from .trainer import move_batch_to_device


class Tester:
    """Patient-level tester for RHDSNet."""

    def __init__(self, config: Dict, ckpt: Optional[str] = None, seed: int = 2024, split: str = "test"):
        self.config = config
        self.seed = int(seed)
        train_cfg = config.get("training", {})
        requested = train_cfg.get("device", "cuda")
        self.device = torch.device("cuda" if requested == "cuda" and torch.cuda.is_available() else "cpu")
        self.output_dir = ensure_dir(Path(train_cfg.get("output_dir", "outputs/rhdsnet")) / f"seed_{self.seed}" / split)
        self.logger = get_logger(f"RHDSNetTester_{self.seed}_{split}", self.output_dir / "test.log")
        self.dataset = DCEMRIDataset(config, split=split, seed=self.seed)
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False, num_workers=0)
        self.model = RHDSNet(config).to(self.device)
        if ckpt:
            load_checkpoint(ckpt, self.model, map_location=self.device)
        self.model.eval()

    @torch.no_grad()
    def run(self):
        rows = []
        threshold = float(self.config.get("inference", {}).get("threshold", 0.5))
        inf_cfg = self.config.get("inference", {})
        roi = crop_size_to_dhw(inf_cfg.get("roi_size"), inf_cfg.get("roi_size_order", "hwd"))
        use_sw = bool(inf_cfg.get("sliding_window", True)) and roi is not None
        for batch in self.loader:
            batch = move_batch_to_device(batch, self.device)
            case_id = batch["case_id"][0] if isinstance(batch["case_id"], list) else batch["case_id"]
            if use_sw:
                logits = sliding_window_inference(
                    self.model,
                    batch["v0"],
                    roi,
                    overlap=float(inf_cfg.get("overlap", 0.5)),
                    sample_steps=self.model.ddim_steps_test,
                )
                outputs = {"logits": logits}
            else:
                outputs = self.model(batch["v0"], sample_steps=self.model.ddim_steps_test)
            pred = logits_to_mask(outputs["logits"], threshold)[0, 0]
            gt = batch["mask"].detach().cpu().numpy()[0, 0]
            spacing = batch["spacing"].detach().cpu().numpy()[0]
            metrics = patient_segmentation_metrics(pred, gt, spacing)
            metrics["case_id"] = case_id

            if self.config.get("metrics", {}).get("save_predictions", True):
                save_volume(pred.astype(np.uint8), self.output_dir / "predictions" / f"{case_id}_pred.npy")
            if self.config.get("metrics", {}).get("save_generated", True):
                gen_out = self.model(batch["v0"], sample_steps=self.model.ddim_steps_test)
                generated = gen_out["generated"].detach().cpu().numpy()[0]
                real = batch["post"].detach().cpu().numpy()[0]
                metrics["mean-SSIM"] = mean_ssim(real, generated)
                save_volume(generated.astype(np.float32), self.output_dir / "generated" / f"{case_id}_generated.npy")
            rows.append(metrics)
            self.logger.info(f"{case_id}: DSC={metrics['DSC'] * 100:.2f} HD95={metrics['HD95']:.2f} Sen={metrics['Sen'] * 100:.2f}")
        summary = summarize_metric_rows(rows, ["DSC", "HD95", "Sen", "mean-SSIM"])
        save_csv(rows, self.output_dir / "patient_metrics.csv")
        save_json({"patients": rows, "summary": summary}, self.output_dir / "metrics.json")
        self.logger.info(f"Summary: {summary}")
        return {"patients": rows, "summary": summary, "output_dir": str(self.output_dir)}
