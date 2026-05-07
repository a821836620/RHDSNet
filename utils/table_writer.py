from pathlib import Path
from typing import Dict, Iterable, List

from .io import save_csv, save_json


def write_table(rows: Iterable[Dict], output_dir, name: str, fieldnames: List[str] = None) -> None:
    """Save experiment tables as CSV and JSON with matching content."""
    output_dir = Path(output_dir)
    rows = list(rows)
    save_csv(rows, output_dir / f"{name}.csv", fieldnames=fieldnames)
    save_json(rows, output_dir / f"{name}.json")
