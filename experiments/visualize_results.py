import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from utils.io import ensure_dir, load_case_npz, load_volume


METHOD_ORDER = ["GT", "RHDSNet", "TSESNet", "DEFNet", "BBDiffSF", "ALMN", "Tumor-sen", "PLHN", "nnUNet"]


def _slice(volume, axis: str, index: Optional[int] = None):
    volume = np.squeeze(volume)
    if index is None:
        index = volume.shape[0 if axis == "axial" else 1 if axis == "coronal" else 2] // 2
    if axis == "axial":
        return volume[index]
    if axis == "coronal":
        return volume[:, index, :]
    if axis == "sagittal":
        return volume[:, :, index]
    raise ValueError(axis)


def _overlay(image2d, mask2d, alpha=0.45):
    image2d = image2d.astype(np.float32)
    image2d = (image2d - image2d.min()) / (image2d.max() - image2d.min() + 1e-6)
    rgb = np.stack([image2d, image2d, image2d], axis=-1)
    mask = mask2d.astype(bool)
    rgb[mask, 0] = (1 - alpha) * rgb[mask, 0] + alpha
    rgb[mask, 1] = (1 - alpha) * rgb[mask, 1]
    rgb[mask, 2] = (1 - alpha) * rgb[mask, 2]
    return rgb


def save_segmentation_comparison(v0, masks: Dict[str, np.ndarray], output_path, axis="axial", index=None, zoom_box=None):
    ensure_dir(Path(output_path).parent)
    methods = [m for m in METHOD_ORDER if m in masks] + [m for m in masks if m not in METHOD_ORDER]
    fig, axes = plt.subplots(1, len(methods), figsize=(3 * len(methods), 3), dpi=160)
    if len(methods) == 1:
        axes = [axes]
    img = _slice(v0, axis, index)
    for ax, method in zip(axes, methods):
        mask = _slice(masks[method], axis, index)
        ax.imshow(_overlay(img, mask), interpolation="nearest")
        if zoom_box:
            y, x, hh, ww = zoom_box
            rect = plt.Rectangle((x, y), ww, hh, fill=False, edgecolor="yellow", linewidth=1.5)
            ax.add_patch(rect)
        ax.set_title(method)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_generation_comparison(real_seq, generated: Dict[str, np.ndarray], output_path, axis="axial", index=None):
    ensure_dir(Path(output_path).parent)
    rows = [("real sequence", real_seq)] + list(generated.items())
    t = real_seq.shape[0]
    fig, axes = plt.subplots(len(rows), t, figsize=(2.2 * t, 2.2 * len(rows)), dpi=160)
    if len(rows) == 1:
        axes = axes[None]
    for r, (name, seq) in enumerate(rows):
        for c in range(t):
            axes[r, c].imshow(_slice(seq[c], axis, index), cmap="gray")
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_ylabel(name)
            axes[r, c].set_title(f"V{c + 1}")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def export_marching_cubes(mask, output_obj, spacing=(1.0, 1.0, 1.0)):
    """Export a simple OBJ surface from a binary mask."""
    ensure_dir(Path(output_obj).parent)
    try:
        from skimage import measure

        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
    except Exception as exc:
        raise ImportError("skimage is required for marching cubes export.") from exc
    with Path(output_obj).open("w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces + 1:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def visualize_from_case(case_npz, pred_dir, output_dir, case_id=None, axis="axial", index=None):
    case = load_case_npz(case_npz, num_post_phases=4)
    case_id = case_id or Path(case_npz).stem
    pred_dir = Path(pred_dir)
    masks = {"GT": case["mask"]}
    for method in METHOD_ORDER:
        pred = pred_dir / method / f"{case_id}_pred.npy"
        if pred.exists():
            masks[method] = load_volume(pred)
    output_dir = ensure_dir(output_dir)
    save_segmentation_comparison(case["V0"], masks, output_dir / f"{case_id}_segmentation_{axis}.png", axis, index)

    generated = {}
    for method in ["RHDSNet", "SPHERE-like", "DCEtriFormer-like"]:
        path = pred_dir / method / f"{case_id}_generated.npy"
        if path.exists():
            generated[method] = load_volume(path)
    if generated:
        save_generation_comparison(case["post"], generated, output_dir / f"{case_id}_generation_{axis}.png", axis, index)
    try:
        export_marching_cubes(case["mask"], output_dir / f"{case_id}_gt_surface.obj")
    except ImportError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Visualize segmentation and DCE generation results.")
    parser.add_argument("--case_npz", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/visualizations")
    parser.add_argument("--case_id", default=None)
    parser.add_argument("--axis", choices=["axial", "sagittal", "coronal"], default="axial")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    visualize_from_case(args.case_npz, args.pred_dir, args.output_dir, args.case_id, args.axis, args.index)


if __name__ == "__main__":
    main()
