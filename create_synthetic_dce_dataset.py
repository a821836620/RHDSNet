import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from utils.io import ensure_dir


def _ellipsoid_mask(shape, center, radii):
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    cz, cy, cx = center
    rz, ry, rx = radii
    dist = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    return dist <= 1.0


def create_synthetic_dataset(
    output_dir,
    num_cases: int = 12,
    num_post_phases: int = 4,
    shape: Tuple[int, int, int] = (48, 96, 96),
    seed: int = 2024,
) -> None:
    """Create a small synthetic DCE-MRI dataset.

    The tumor region gradually enhances over post-contrast phases while the
    background stays approximately stable. Files are keyed NPZ cases containing
    V0, V1...VT, post, mask, and spacing.
    """
    output_dir = ensure_dir(output_dir)
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, shape[0]),
        np.linspace(-1, 1, shape[1]),
        np.linspace(-1, 1, shape[2]),
        indexing="ij",
    )
    breast_like = np.exp(-((yy / 0.85) ** 2 + (xx / 0.65) ** 2)) * (np.abs(zz) < 0.95)

    for idx in range(num_cases):
        center = (
            rng.integers(shape[0] // 4, 3 * shape[0] // 4),
            rng.integers(shape[1] // 4, 3 * shape[1] // 4),
            rng.integers(shape[2] // 4, 3 * shape[2] // 4),
        )
        radii = (
            rng.integers(max(3, shape[0] // 14), max(5, shape[0] // 7)),
            rng.integers(max(5, shape[1] // 14), max(9, shape[1] // 7)),
            rng.integers(max(5, shape[2] // 14), max(9, shape[2] // 7)),
        )
        mask = _ellipsoid_mask(shape, center, radii).astype(np.float32)
        texture = 0.08 * rng.normal(size=shape).astype(np.float32)
        bias = 0.15 * (zz + 1.0).astype(np.float32)
        v0 = (0.55 * breast_like + bias + texture).astype(np.float32)
        post = []
        for phase in range(1, num_post_phases + 1):
            enhancement = (0.18 + 0.13 * phase) * mask
            rim = (0.08 * phase) * (_ellipsoid_mask(shape, center, tuple(max(1, r + 2) for r in radii)).astype(np.float32) - mask)
            background_drift = 0.015 * phase * breast_like
            noise = 0.05 * rng.normal(size=shape).astype(np.float32)
            post.append((v0 + enhancement + rim + background_drift + noise).astype(np.float32))
        payload = {
            "V0": v0.astype(np.float32),
            "post": np.stack(post, axis=0).astype(np.float32),
            "mask": mask.astype(np.float32),
            "spacing": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        }
        for phase, vol in enumerate(post, start=1):
            payload[f"V{phase}"] = vol.astype(np.float32)
        np.savez_compressed(output_dir / f"case_{idx:03d}.npz", **payload)


def main():
    parser = argparse.ArgumentParser(description="Create a synthetic DCE-MRI breast tumor dataset.")
    parser.add_argument("--output_dir", default="data/synthetic_ispy2")
    parser.add_argument("--num_cases", type=int, default=12)
    parser.add_argument("--num_post_phases", type=int, default=4)
    parser.add_argument("--shape", type=int, nargs=3, default=[48, 96, 96], help="D H W")
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    create_synthetic_dataset(args.output_dir, args.num_cases, args.num_post_phases, tuple(args.shape), args.seed)
    print(f"Saved synthetic dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
