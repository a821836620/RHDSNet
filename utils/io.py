import csv
import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

try:
    import nibabel as nib
except ImportError:  # pragma: no cover
    nib = None


def ensure_dir(path: os.PathLike) -> Path:
    """Create a directory if needed and return it as Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating inputs."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: os.PathLike) -> Dict[str, Any]:
    """Load YAML config and recursively resolve optional base_config."""
    if yaml is None:
        raise ImportError("PyYAML is required to load YAML configs.")
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("base_config", None)
    if base_ref:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
            if not base_path.exists():
                base_path = Path(base_ref)
        base_cfg = load_config(base_path)
        cfg = deep_update(base_cfg, cfg)
    cfg["_config_path"] = str(path)
    return cfg


def save_config(config: Dict[str, Any], path: os.PathLike) -> None:
    """Save a config snapshot as YAML if available, otherwise JSON."""
    ensure_dir(Path(path).parent)
    serializable = {k: v for k, v in config.items() if not k.startswith("_")}
    if yaml is not None:
        with Path(path).open("w", encoding="utf-8") as f:
            yaml.safe_dump(serializable, f, sort_keys=False)
    else:  # pragma: no cover
        save_json(serializable, path)


def copy_config(config_path: Optional[str], output_dir: os.PathLike) -> None:
    if config_path and Path(config_path).exists():
        ensure_dir(output_dir)
        shutil.copy2(config_path, Path(output_dir) / Path(config_path).name)


def save_json(obj: Any, path: os.PathLike) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: os.PathLike) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(rows: Iterable[Dict[str, Any]], path: os.PathLike, fieldnames: Optional[List[str]] = None) -> None:
    rows = list(rows)
    ensure_dir(Path(path).parent)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_volume(path: os.PathLike) -> np.ndarray:
    """Load a 3D/4D volume from nii.gz, nii, npy, or npz."""
    path = Path(path)
    name = path.name.lower()
    if name.endswith(".npy"):
        return np.load(path).astype(np.float32)
    if name.endswith(".npz"):
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"].astype(np.float32)
        raise ValueError(f"NPZ volume {path} has no arr_0 key; use load_case_npz for keyed cases.")
    if name.endswith(".nii") or name.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel is required for NIfTI files.")
        return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32), dtype=np.float32)
    raise ValueError(f"Unsupported volume format: {path}")


def load_case_npz(path: os.PathLike, num_post_phases: int) -> Dict[str, np.ndarray]:
    """Load a keyed NPZ case containing V0, V1...VT, and mask."""
    data = np.load(path)
    case = {"V0": data["V0"].astype(np.float32)}
    phases = []
    for i in range(1, num_post_phases + 1):
        key = f"V{i}"
        if key in data:
            phases.append(data[key].astype(np.float32))
    if "post" in data:
        post = data["post"].astype(np.float32)
        if post.ndim == 4:
            post_t = post if post.shape[0] <= num_post_phases + 1 else np.moveaxis(post, -1, 0)
            phases = [post_t[i] for i in range(min(post_t.shape[0], num_post_phases))]
    if not phases:
        raise ValueError(f"Case NPZ {path} has no post-contrast phases.")
    mask_key = "mask" if "mask" in data else "GT"
    case["post"] = np.stack(phases, axis=0).astype(np.float32)
    case["mask"] = data[mask_key].astype(np.float32)
    if "spacing" in data:
        case["spacing"] = data["spacing"].astype(np.float32)
    return case


def save_volume(array: np.ndarray, path: os.PathLike, reference: Optional[os.PathLike] = None) -> None:
    """Save a 3D/4D volume. NIfTI saving is optional and uses reference affine when given."""
    path = Path(path)
    ensure_dir(path.parent)
    name = path.name.lower()
    array = np.asarray(array)
    if name.endswith(".npy"):
        np.save(path, array)
    elif name.endswith(".npz"):
        np.savez_compressed(path, arr_0=array)
    elif name.endswith(".nii") or name.endswith(".nii.gz"):
        if nib is None:
            raise ImportError("nibabel is required for NIfTI saving.")
        affine = np.eye(4)
        if reference is not None and Path(reference).exists():
            affine = nib.load(str(reference)).affine
        nib.save(nib.Nifti1Image(array.astype(np.float32), affine), str(path))
    else:
        raise ValueError(f"Unsupported output format: {path}")


def format_mean_std(values: Iterable[float], scale: float = 1.0, precision: int = 2) -> str:
    arr = np.asarray(list(values), dtype=np.float64) * scale
    if arr.size == 0:
        return "nan"
    return f"{arr.mean():.{precision}f} ± {arr.std(ddof=1 if arr.size > 1 else 0):.{precision}f}"


def crop_size_to_dhw(size, order: str = "hwd") -> List[int]:
    """Convert config crop/roi size into internal D,H,W order."""
    if size is None:
        return None
    size = list(size)
    if order.lower() == "dhw":
        return [int(size[0]), int(size[1]), int(size[2])]
    if order.lower() == "hwd":
        return [int(size[2]), int(size[0]), int(size[1])]
    raise ValueError(f"Unknown size order: {order}")
