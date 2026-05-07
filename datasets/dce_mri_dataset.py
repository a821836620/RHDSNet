import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.io import crop_size_to_dhw, load_case_npz, load_volume
from .transforms import (
    center_crop_sample,
    patient_splits,
    preprocess_sequence,
    random_crop_sample,
    random_flip_sample,
    random_intensity_scale,
)


SUPPORTED_SUFFIXES = (".nii.gz", ".nii", ".npy", ".npz")


def _is_supported(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in SUPPORTED_SUFFIXES)


def _find_named_file(case_dir: Path, names: List[str]) -> Optional[Path]:
    """Find a file whose stem-like prefix matches one of names."""
    files = [p for p in case_dir.iterdir() if p.is_file() and _is_supported(p)]
    for target in names:
        target_low = target.lower()
        for p in files:
            low = p.name.lower()
            if low == target_low or low.startswith(target_low + ".") or low.startswith(target_low + "_"):
                return p
    return None


def discover_cases(root, num_post_phases: int, manifest: Optional[str] = None) -> List[Dict]:
    """Discover DCE-MRI cases from a manifest CSV, keyed NPZ files, or case folders."""
    root = Path(root)
    cases = []
    if manifest:
        def resolve(value):
            if value is None or value == "":
                return value
            path = Path(value)
            return str(path if path.is_absolute() else root / path)

        with Path(manifest).open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                case = {
                    "case_id": row.get("case_id") or row.get("id"),
                    "V0": resolve(row["V0"]),
                    "mask": resolve(row.get("mask") or row.get("GT")),
                    "post": [resolve(row[f"V{i}"]) for i in range(1, num_post_phases + 1) if row.get(f"V{i}")],
                    "spacing": row.get("spacing"),
                }
                cases.append(case)
        return cases

    if not root.exists():
        return []

    for npz in sorted(root.glob("*.npz")):
        try:
            data = np.load(npz)
            if "V0" in data and ("mask" in data or "GT" in data):
                cases.append({"case_id": npz.stem, "npz": str(npz)})
        except Exception:
            continue

    for case_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        v0 = _find_named_file(case_dir, ["V0", "pre", "pre_contrast", "precontrast"])
        mask = _find_named_file(case_dir, ["mask", "GT", "label", "seg"])
        sequence = _find_named_file(case_dir, ["sequence", "dce", "4d"])
        post = []
        for i in range(1, num_post_phases + 1):
            p = _find_named_file(case_dir, [f"V{i}", f"post{i}", f"phase{i}"])
            if p is not None:
                post.append(str(p))
        if sequence is not None and mask is not None:
            cases.append({"case_id": case_dir.name, "sequence": str(sequence), "mask": str(mask)})
        elif v0 is not None and mask is not None and post:
            cases.append({"case_id": case_dir.name, "V0": str(v0), "post": post, "mask": str(mask)})
    return cases


class DCEMRIDataset(Dataset):
    """Patient-level DCE-MRI dataset.

    Returned tensors use shape:
      v0:   [1, D, H, W]
      post: [T, D, H, W]
      mask: [1, D, H, W]
    """

    def __init__(self, config: Dict, split: str = "train", seed: int = 2024, case_ids: Optional[List[str]] = None):
        self.config = config
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.dataset_cfg = config.get("dataset", {})
        self.pre_cfg = config.get("preprocess", {})
        self.num_post = int(self.dataset_cfg.get("num_post_phases", config.get("model", {}).get("num_post_phases", 4)))
        self.root = Path(self.dataset_cfg.get("root", "data"))
        self.crop_dhw = crop_size_to_dhw(
            self.pre_cfg.get("crop_size"),
            self.pre_cfg.get("crop_size_order", "hwd"),
        )

        self._maybe_create_synthetic()
        all_cases = discover_cases(self.root, self.num_post, self.dataset_cfg.get("manifest"))
        if not all_cases:
            raise FileNotFoundError(f"No DCE-MRI cases found in {self.root}.")
        self.case_map = {c["case_id"]: c for c in all_cases}
        if case_ids is None:
            split_ids = self._load_or_create_splits(list(self.case_map.keys()), seed)
            case_ids = split_ids[split]
        self.cases = [self.case_map[cid] for cid in case_ids if cid in self.case_map]
        if not self.cases:
            raise ValueError(f"Split {split} contains no valid cases.")

    def _maybe_create_synthetic(self) -> None:
        if self.root.exists() and any(self.root.iterdir()):
            return
        if not self.dataset_cfg.get("synthetic_if_missing", False):
            return
        from create_synthetic_dce_dataset import create_synthetic_dataset

        create_synthetic_dataset(
            output_dir=self.root,
            num_cases=12,
            num_post_phases=self.num_post,
            shape=(48, 96, 96),
            seed=self.seed,
        )

    def _load_or_create_splits(self, case_ids: List[str], seed: int) -> Dict[str, List[str]]:
        split_file = self.dataset_cfg.get("split_file")
        if split_file and Path(split_file).exists():
            with Path(split_file).open("r", encoding="utf-8") as f:
                return json.load(f)
        return patient_splits(
            case_ids,
            self.dataset_cfg.get("train_ratio", 0.7),
            self.dataset_cfg.get("val_ratio", 0.1),
            self.dataset_cfg.get("test_ratio", 0.2),
            seed=seed,
        )

    def __len__(self):
        return len(self.cases)

    def set_epoch(self, epoch: int) -> None:
        """Update deterministic augmentation seed for the next training epoch."""
        self.epoch = int(epoch)

    def _load_case(self, case: Dict) -> Dict:
        if "npz" in case:
            loaded = load_case_npz(case["npz"], self.num_post)
            v0, post, mask = loaded["V0"], loaded["post"], loaded["mask"]
            spacing = loaded.get("spacing", self.dataset_cfg.get("spacing", [1.0, 1.0, 1.0]))
            reference = case["npz"]
        elif "sequence" in case:
            seq = load_volume(case["sequence"])
            if seq.ndim != 4:
                raise ValueError(f"Expected 4D sequence for {case['case_id']}, got {seq.shape}")
            # Accept both [T, D, H, W] and [D, H, W, T].
            if seq.shape[0] <= self.num_post + 1:
                seq_t = seq
            else:
                seq_t = np.moveaxis(seq, -1, 0)
            v0 = seq_t[0]
            post = seq_t[1 : self.num_post + 1]
            mask = load_volume(case["mask"])
            spacing = self.dataset_cfg.get("spacing", [1.0, 1.0, 1.0])
            reference = case["sequence"]
        else:
            v0 = load_volume(case["V0"])
            post = np.stack([load_volume(p) for p in case["post"][: self.num_post]], axis=0)
            mask = load_volume(case["mask"])
            spacing = self.dataset_cfg.get("spacing", [1.0, 1.0, 1.0])
            reference = case["V0"]
        if mask.ndim == 4:
            mask = np.squeeze(mask)
        mask = (mask > 0).astype(np.float32)
        v0, post = preprocess_sequence(
            v0,
            post,
            clip_pct=float(self.pre_cfg.get("clip_percentile", 0.1)),
            normalize=self.pre_cfg.get("normalize", "zscore"),
        )
        return {
            "case_id": case["case_id"],
            "v0": v0.astype(np.float32),
            "post": post.astype(np.float32),
            "mask": mask.astype(np.float32),
            "spacing": np.asarray(spacing, dtype=np.float32),
            "reference": reference,
            "original_shape": np.asarray(v0.shape[-3:], dtype=np.int32),
        }

    def __getitem__(self, index):
        case = self.cases[index]
        sample = self._load_case(case)
        rng = np.random.default_rng(self.seed + index + self.epoch * max(len(self.cases), 1) + (0 if self.split == "train" else 100000))
        if self.crop_dhw and self.split == "train":
            sample = random_crop_sample(sample, self.crop_dhw, rng)
            if self.pre_cfg.get("random_flip", True):
                sample = random_flip_sample(sample, rng)
            if self.pre_cfg.get("random_intensity_scale", True):
                sample = random_intensity_scale(sample, rng)
        elif self.crop_dhw and self.split == "val":
            sample = center_crop_sample(sample, self.crop_dhw)

        v0 = torch.from_numpy(sample["v0"][None]).float()
        post = torch.from_numpy(sample["post"]).float()
        mask = torch.from_numpy(sample["mask"][None]).float()
        return {
            "case_id": sample["case_id"],
            "v0": v0,
            "post": post,
            "mask": mask,
            "spacing": torch.from_numpy(sample["spacing"]).float(),
            "reference": sample["reference"],
            "original_shape": torch.from_numpy(sample["original_shape"]),
        }
