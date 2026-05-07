from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from models.baselines.fourd_unet import FourDUNetProxy
from models.baselines.generate_then_segment import NamedGeneratorProxy
from models.baselines.simple_diffusion_generator import DDPMGeneratorProxy
from models.rhdsnet import RHDSNet
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.io import ensure_dir
from utils.table_writer import write_table
from .common import build_loaders, evaluate_segmentation_model, get_device, move_batch, train_segmentation_proxy


GENERATOR_NAMES = [
    "TeNCA-like generator",
    "DCEtriFormer-like generator",
    "SPHERE-like generator",
    "DDPM generator baseline",
]


class FrozenSegmentWithGenerator(nn.Module):
    """Feed generated phases into the same full-sequence TSESNet proxy."""

    def __init__(self, generator: nn.Module, segmenter: nn.Module):
        super().__init__()
        self.generator = generator
        self.segmenter = segmenter

    def forward(self, batch):
        v0 = batch["v0"] if isinstance(batch, dict) else batch[:, :1]
        post_hat = self.generator(v0)
        logits = self.segmenter({"v0": v0, "post": post_hat})["logits"]
        return {"logits": logits, "generated": post_hat}


def train_generator(config: Dict, generator: nn.Module, output_dir, seed: int = 2024):
    device = get_device(config)
    train_loader, _, _ = build_loaders(config, seed)
    generator = generator.to(device)
    opt = torch.optim.Adam(generator.parameters(), lr=float(config.get("training", {}).get("lr", 1e-4)))
    scaler = GradScaler(enabled=bool(config.get("training", {}).get("amp", True)) and device.type == "cuda")
    epochs = int(config.get("experiment", {}).get("generator_epochs", config.get("experiment", {}).get("proxy_epochs", 50)))
    history = []
    for epoch in range(epochs):
        generator.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        total = 0.0
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                pred = generator(batch["v0"])
                loss = F.mse_loss(pred, batch["post"])
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            count += 1
        history.append({"epoch": epoch, "mse": total / max(count, 1)})
    save_checkpoint(Path(output_dir) / "generator.pth", generator, opt, epoch=epochs - 1, best_metric=0.0, config=config)
    return generator


def run_generate_then_segment(config: Dict):
    output_dir = ensure_dir(config.get("experiment", {}).get("output_dir", Path(config.get("training", {}).get("output_dir", "outputs")) / "generate_then_segment"))
    seed = config.get("training", {}).get("seeds", [2024])[0]
    device = get_device(config)
    _, _, test_loader = build_loaders(config, seed)

    oracle_dir = ensure_dir(Path(output_dir) / "oracle_tsesnet")
    oracle = train_segmentation_proxy(config, FourDUNetProxy(config), oracle_dir, seed=seed)
    oracle = oracle.to(device).eval()
    rows: List[Dict] = []
    oracle_eval = evaluate_segmentation_model(config, oracle, test_loader, device, save_dir=oracle_dir)
    rows.append(
        {
            "Methods": "TSESNet oracle full-sequence",
            "mean-SSIM": 1.0,
            "DSC": oracle_eval["summary"].get("DSC_mean", 0.0) * 100,
            "HD95": oracle_eval["summary"].get("HD95_mean", 0.0),
            "Sen": oracle_eval["summary"].get("Sen_mean", 0.0) * 100,
        }
    )

    for name in GENERATOR_NAMES:
        gen_dir = ensure_dir(Path(output_dir) / name.replace(" ", "_"))
        if "DDPM" in name:
            gen = DDPMGeneratorProxy(
                int(config.get("model", {}).get("num_post_phases", 4)),
                int(config.get("model", {}).get("base_channels", 24)),
            )
        else:
            gen = NamedGeneratorProxy(config, name)
        gen = train_generator(config, gen, gen_dir, seed=seed)
        wrapper = FrozenSegmentWithGenerator(gen, oracle).to(device)
        result = evaluate_segmentation_model(config, wrapper, test_loader, device, save_dir=gen_dir, generated_from_output=True)
        rows.append(
            {
                "Methods": name,
                "mean-SSIM": result["summary"].get("mean-SSIM_mean", 0.0),
                "DSC": result["summary"].get("DSC_mean", 0.0) * 100,
                "HD95": result["summary"].get("HD95_mean", 0.0),
                "Sen": result["summary"].get("Sen_mean", 0.0) * 100,
            }
        )

    rhdsnet = RHDSNet(config).to(device)
    ckpt = Path(config.get("training", {}).get("output_dir", "outputs/rhdsnet")) / f"seed_{seed}" / "best.pth"
    if ckpt.exists():
        load_checkpoint(ckpt, rhdsnet, map_location=device)
    rhdsnet_result = evaluate_segmentation_model(config, rhdsnet, test_loader, device, save_dir=Path(output_dir) / "RHDSNet", generated_from_output=True)
    rows.append(
        {
            "Methods": "RHDSNet",
            "mean-SSIM": rhdsnet_result["summary"].get("mean-SSIM_mean", 0.0),
            "DSC": rhdsnet_result["summary"].get("DSC_mean", 0.0) * 100,
            "HD95": rhdsnet_result["summary"].get("HD95_mean", 0.0),
            "Sen": rhdsnet_result["summary"].get("Sen_mean", 0.0) * 100,
        }
    )
    write_table(rows, output_dir, "generate_then_segment", ["Methods", "mean-SSIM", "DSC", "HD95", "Sen"])
    return rows
