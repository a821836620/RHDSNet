from itertools import product
from typing import Sequence

import torch
import torch.nn.functional as F


def _compute_starts(dim: int, roi: int, overlap: float):
    if dim <= roi:
        return [0]
    stride = max(1, int(roi * (1.0 - overlap)))
    starts = list(range(0, dim - roi + 1, stride))
    if starts[-1] != dim - roi:
        starts.append(dim - roi)
    return starts


@torch.no_grad()
def sliding_window_inference(model, v0: torch.Tensor, roi_size_dhw: Sequence[int], overlap: float = 0.5, **model_kwargs):
    """Sliding-window inference for V0-only RHDSNet.

    v0 shape: [B, 1, D, H, W]. Batch size 1 is recommended for 3D medical volumes.
    """
    if v0.shape[0] != 1:
        raise ValueError("sliding_window_inference currently expects batch size 1.")
    _, _, d, h, w = v0.shape
    rd, rh, rw = [int(x) for x in roi_size_dhw]
    pd, ph, pw = max(0, rd - d), max(0, rh - h), max(0, rw - w)
    if pd or ph or pw:
        v0_pad = F.pad(v0, (0, pw, 0, ph, 0, pd))
    else:
        v0_pad = v0
    _, _, dd, hh, ww = v0_pad.shape
    output = torch.zeros((1, 1, dd, hh, ww), device=v0.device, dtype=v0.dtype)
    count = torch.zeros_like(output)
    d_starts = _compute_starts(dd, rd, overlap)
    h_starts = _compute_starts(hh, rh, overlap)
    w_starts = _compute_starts(ww, rw, overlap)
    for sd, sh, sw in product(d_starts, h_starts, w_starts):
        patch = v0_pad[:, :, sd : sd + rd, sh : sh + rh, sw : sw + rw]
        logits = model(patch, **model_kwargs)["logits"]
        output[:, :, sd : sd + rd, sh : sh + rh, sw : sw + rw] += logits
        count[:, :, sd : sd + rd, sh : sh + rh, sw : sw + rw] += 1
    output = output / torch.clamp(count, min=1)
    return output[:, :, :d, :h, :w]
