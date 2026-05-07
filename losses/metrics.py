from typing import Dict, Iterable, List, Optional

import numpy as np
import torch


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = np.logical_and(pred, target).sum()
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * inter + eps) / (denom + eps))


def sensitivity_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = np.logical_and(pred, target).sum()
    fn = np.logical_and(~pred, target).sum()
    if target.sum() == 0:
        return 1.0
    return float((tp + eps) / (tp + fn + eps))


def hd95_score(pred: np.ndarray, target: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> float:
    """95% Hausdorff distance in mm. Returns 0 when both masks are empty and inf for one-empty cases."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if pred.sum() == 0 and target.sum() == 0:
        return 0.0
    if pred.sum() == 0 or target.sum() == 0:
        return float("inf")
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt

        pred_border = np.logical_xor(pred, binary_erosion(pred))
        target_border = np.logical_xor(target, binary_erosion(target))
        dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
        dt_target = distance_transform_edt(~target_border, sampling=spacing)
        dists = np.concatenate([dt_target[pred_border], dt_pred[target_border]])
        return float(np.percentile(dists, 95))
    except Exception:
        pred_pts = np.argwhere(pred)
        target_pts = np.argwhere(target)
        if pred_pts.shape[0] * target_pts.shape[0] > 5_000_000:
            return float("nan")
        diff = (pred_pts[:, None, :] - target_pts[None, :, :]) * np.asarray(spacing)[None, None, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        mins = np.concatenate([dist.min(axis=1), dist.min(axis=0)])
        return float(np.percentile(mins, 95))


def mean_ssim(real: np.ndarray, pred: np.ndarray) -> float:
    """Mean slice-wise SSIM for volumes or phase sequences."""
    real = np.asarray(real, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    if real.shape != pred.shape:
        raise ValueError(f"SSIM shape mismatch: {real.shape} vs {pred.shape}")
    try:
        from skimage.metrics import structural_similarity

        if real.ndim == 3:
            vals = []
            for z in range(real.shape[0]):
                r, p = real[z], pred[z]
                data_range = float(max(r.max(), p.max()) - min(r.min(), p.min()) + 1e-6)
                vals.append(structural_similarity(r, p, data_range=data_range))
            return float(np.mean(vals))
        if real.ndim == 4:
            return float(np.mean([mean_ssim(real[i], pred[i]) for i in range(real.shape[0])]))
    except Exception:
        mse = np.mean((real - pred) ** 2)
        var = np.var(real) + 1e-6
        return float(max(0.0, 1.0 - mse / var))
    raise ValueError(f"Unsupported SSIM ndim: {real.ndim}")


def logits_to_mask(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    prob = torch.sigmoid(logits).detach().cpu().numpy()
    return (prob >= threshold).astype(np.uint8)


def patient_segmentation_metrics(pred_mask, gt_mask, spacing=(1.0, 1.0, 1.0)) -> Dict[str, float]:
    """Compute patient-level DSC/HD95/Sensitivity. DSC and Sen are fractions."""
    pred = np.squeeze(np.asarray(pred_mask))
    gt = np.squeeze(np.asarray(gt_mask))
    return {
        "DSC": dice_score(pred, gt),
        "HD95": hd95_score(pred, gt, spacing),
        "Sen": sensitivity_score(pred, gt),
    }


def summarize_metric_rows(rows: Iterable[Dict[str, float]], metric_keys: Optional[List[str]] = None) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    metric_keys = metric_keys or [k for k, v in rows[0].items() if isinstance(v, (int, float))]
    summary = {}
    for key in metric_keys:
        values = np.asarray([r[key] for r in rows if key in r and np.isfinite(r[key])], dtype=np.float64)
        if values.size == 0:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
        else:
            summary[f"{key}_mean"] = float(values.mean())
            summary[f"{key}_std"] = float(values.std(ddof=1 if values.size > 1 else 0))
    return summary
