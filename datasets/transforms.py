from typing import Dict, List, Sequence, Tuple

import numpy as np


def clip_percentiles(volume: np.ndarray, percentile: float = 0.1) -> np.ndarray:
    """Clip bottom/top percentile for one volume."""
    if percentile <= 0:
        return volume.astype(np.float32)
    lo = np.percentile(volume, percentile)
    hi = np.percentile(volume, 100.0 - percentile)
    return np.clip(volume, lo, hi).astype(np.float32)


def normalize_volume(volume: np.ndarray, mode: str = "zscore", eps: float = 1e-6) -> np.ndarray:
    """Normalize one 3D volume either by z-score or min-max."""
    volume = volume.astype(np.float32)
    if mode == "none":
        return volume
    if mode == "zscore":
        return (volume - volume.mean()) / (volume.std() + eps)
    if mode in ("minmax", "0-1", "01"):
        vmin, vmax = volume.min(), volume.max()
        return (volume - vmin) / (vmax - vmin + eps)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def preprocess_sequence(v0: np.ndarray, post: np.ndarray, clip_pct: float, normalize: str) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess V0 and each post phase independently."""
    v0 = normalize_volume(clip_percentiles(v0, clip_pct), normalize)
    post_out = []
    for i in range(post.shape[0]):
        post_out.append(normalize_volume(clip_percentiles(post[i], clip_pct), normalize))
    return v0.astype(np.float32), np.stack(post_out, axis=0).astype(np.float32)


def _pad_to_shape(array: np.ndarray, target_dhw: Sequence[int], is_post: bool = False) -> np.ndarray:
    """Pad 3D or T,3D arrays to at least target D,H,W."""
    spatial = array.shape[-3:]
    pad_width = []
    if is_post:
        pad_width.append((0, 0))
    for dim, target in zip(spatial, target_dhw):
        need = max(0, int(target) - int(dim))
        before = need // 2
        after = need - before
        pad_width.append((before, after))
    if any(p[0] or p[1] for p in pad_width[-3:]):
        array = np.pad(array, pad_width, mode="constant")
    return array


def random_crop_sample(sample: Dict, crop_dhw: Sequence[int], rng: np.random.Generator) -> Dict:
    """Randomly crop v0/post/mask with shared coordinates in D,H,W order."""
    v0 = _pad_to_shape(sample["v0"], crop_dhw)
    post = _pad_to_shape(sample["post"], crop_dhw, is_post=True)
    mask = _pad_to_shape(sample["mask"], crop_dhw)
    d, h, w = v0.shape[-3:]
    cd, ch, cw = [int(x) for x in crop_dhw]
    sd = int(rng.integers(0, d - cd + 1)) if d > cd else 0
    sh = int(rng.integers(0, h - ch + 1)) if h > ch else 0
    sw = int(rng.integers(0, w - cw + 1)) if w > cw else 0
    slices = (slice(sd, sd + cd), slice(sh, sh + ch), slice(sw, sw + cw))
    sample = dict(sample)
    sample["v0"] = v0[slices]
    sample["post"] = post[(slice(None),) + slices]
    sample["mask"] = mask[slices]
    return sample


def center_crop_sample(sample: Dict, crop_dhw: Sequence[int]) -> Dict:
    """Center crop v0/post/mask with shared coordinates in D,H,W order."""
    v0 = _pad_to_shape(sample["v0"], crop_dhw)
    post = _pad_to_shape(sample["post"], crop_dhw, is_post=True)
    mask = _pad_to_shape(sample["mask"], crop_dhw)
    d, h, w = v0.shape[-3:]
    cd, ch, cw = [int(x) for x in crop_dhw]
    sd, sh, sw = max(0, (d - cd) // 2), max(0, (h - ch) // 2), max(0, (w - cw) // 2)
    slices = (slice(sd, sd + cd), slice(sh, sh + ch), slice(sw, sw + cw))
    sample = dict(sample)
    sample["v0"] = v0[slices]
    sample["post"] = post[(slice(None),) + slices]
    sample["mask"] = mask[slices]
    return sample


def random_flip_sample(sample: Dict, rng: np.random.Generator, prob: float = 0.5) -> Dict:
    """Apply shared random flips over D/H/W axes."""
    sample = dict(sample)
    for axis in range(3):
        if rng.random() < prob:
            sample["v0"] = np.flip(sample["v0"], axis=axis).copy()
            sample["post"] = np.flip(sample["post"], axis=axis + 1).copy()
            sample["mask"] = np.flip(sample["mask"], axis=axis).copy()
    return sample


def random_intensity_scale(sample: Dict, rng: np.random.Generator, prob: float = 0.5) -> Dict:
    """Apply mild image-only intensity jitter."""
    if rng.random() >= prob:
        return sample
    sample = dict(sample)
    scale = float(rng.uniform(0.9, 1.1))
    shift = float(rng.uniform(-0.05, 0.05))
    sample["v0"] = (sample["v0"] * scale + shift).astype(np.float32)
    sample["post"] = (sample["post"] * scale + shift).astype(np.float32)
    return sample


def patient_splits(case_ids: List[str], train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=2024):
    """Return deterministic patient-level train/val/test case id lists."""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.")
    rng = np.random.default_rng(seed)
    ids = list(case_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    if not test and ids:
        test = [ids[-1]]
        train = train[:-1] if train else train
    return {"train": train, "val": val, "test": test}
