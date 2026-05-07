import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from engine.evaluator import confidence_interval
from utils.io import ensure_dir
from utils.table_writer import write_table


DEFAULT_BASELINES = ["TSESNet", "BBDiffSF", "DEFNet"]


def _read_metric_csv(path: Path, metric: str) -> List[float]:
    values = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if metric in row and row[metric] not in ("", "nan", "None"):
                values.append(float(row[metric]))
    return values


def _find_method_files(result_dir: Path, method: str) -> List[Path]:
    candidates = list(result_dir.rglob("patient_metrics.csv"))
    method_low = method.lower()
    matches = sorted([path for path in candidates if method_low in str(path).lower()])
    if not matches:
        raise FileNotFoundError(f"No patient_metrics.csv found for method {method} under {result_dir}")
    return matches


def _read_metric_files(paths: List[Path], metric: str) -> List[float]:
    values = []
    for path in paths:
        values.extend(_read_metric_csv(path, metric))
    return values


def _paired_ttest(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    diff = a - b
    try:
        from scipy import stats

        p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
    except Exception:
        if n < 2:
            p = float("nan")
        else:
            se = diff.std(ddof=1) / np.sqrt(n)
            t = diff.mean() / (se + 1e-12)
            p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t) / np.sqrt(2.0)))))
    return diff, p


def _mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return "nan"
    return f"{values.mean():.4f} ± {values.std(ddof=1 if values.size > 1 else 0):.4f}"


def run_statistical_test(result_dir: str, dataset: str = "ISPY2", baselines: Iterable[str] = None, metrics: Iterable[str] = None):
    """Paired t-test between RHDSNet and selected baselines from saved patient CSV files."""
    result_dir = Path(result_dir)
    baselines = list(baselines or DEFAULT_BASELINES)
    metrics = list(metrics or ["DSC", "HD95", "Sen"])
    rhds_files = _find_method_files(result_dir, "RHDSNet")
    rows: List[Dict] = []
    for baseline in baselines:
        try:
            base_files = _find_method_files(result_dir, baseline)
        except FileNotFoundError:
            continue
        for metric in metrics:
            rhds = _read_metric_files(rhds_files, metric)
            base = _read_metric_files(base_files, metric)
            diff, p = _paired_ttest(rhds, base)
            ci_low, ci_high = confidence_interval(diff)
            rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Baseline": baseline,
                    "RHDSNet mean ± std": _mean_std(rhds),
                    "Baseline mean ± std": _mean_std(base),
                    "95% CI of paired diff": f"[{ci_low:.4f}, {ci_high:.4f}]",
                    "p-value": p,
                }
            )
    out_dir = ensure_dir(result_dir / "statistics")
    write_table(rows, out_dir, "paired_t_test", ["Dataset", "Metric", "Baseline", "RHDSNet mean ± std", "Baseline mean ± std", "95% CI of paired diff", "p-value"])
    return rows
