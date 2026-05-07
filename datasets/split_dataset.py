import argparse
import json
from pathlib import Path

from datasets.dce_mri_dataset import discover_cases
from datasets.transforms import patient_splits
from utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="Create patient-level train/val/test splits.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_post_phases", type=int, default=4)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    cases = discover_cases(args.root, args.num_post_phases)
    splits = patient_splits(
        [c["case_id"] for c in cases],
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )
    ensure_dir(Path(args.output).parent)
    with Path(args.output).open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {args.output}")


if __name__ == "__main__":
    main()
