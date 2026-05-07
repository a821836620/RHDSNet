from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from losses.metrics import summarize_metric_rows
from utils.io import load_json, save_json


def aggregate_five_runs(result_dirs: Iterable[str], metric_keys: List[str] = None) -> Dict:
    """Aggregate result JSON files from independent seeds."""
    metric_keys = metric_keys or ["DSC", "HD95", "Sen", "mean-SSIM"]
    summaries = []
    for d in result_dirs:
        path = Path(d) / "metrics.json"
        if path.exists():
            summaries.append(load_json(path)["summary"])
    rows = []
    for summary in summaries:
        row = {}
        for key in metric_keys:
            mean_key = f"{key}_mean"
            if mean_key in summary:
                row[key] = summary[mean_key]
        rows.append(row)
    return summarize_metric_rows(rows, metric_keys)


def confidence_interval(values, confidence: float = 0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return (float("nan"), float("nan"))
    mean = values.mean()
    if values.size == 1:
        return (float(mean), float(mean))
    try:
        from scipy import stats

        sem = stats.sem(values)
        h = sem * stats.t.ppf((1 + confidence) / 2.0, values.size - 1)
        return float(mean - h), float(mean + h)
    except Exception:
        h = 1.96 * values.std(ddof=1) / np.sqrt(values.size)
        return float(mean - h), float(mean + h)


def save_aggregate(summary: Dict, output_path):
    save_json(summary, output_path)
