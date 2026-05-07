# RHDSNet

PyTorch implementation of **RHDSNet: Residual Hemodynamic Dynamics Distillation for Breast Tumor Segmentation from Incomplete DCE-MRI with Only a Single Pre-contrast MRI**.

The project trains a recurrent conditional diffusion module (RHDD) to distill post-contrast hemodynamic dynamics from `V0`, then segments breast tumor from **only one pre-contrast MRI at inference time**. Real post-contrast phases are used only during training for diffusion supervision.

## Environment

```bash
pip install -r requirements.txt
```

Optional packages:

- `nibabel` for `.nii/.nii.gz`
- `scipy` for HD95 and paired t-test
- `scikit-image` for SSIM and marching-cubes visualization
- `thop` for FLOPs

## Data Format

Each case can be stored as a keyed `.npz`:

```text
case_000.npz
  V0:   [D, H, W]
  V1:   [D, H, W]
  V2:   [D, H, W]
  ...
  post: [T, D, H, W]
  mask: [D, H, W]
  spacing: [3]
```

Or as one folder per patient:

```text
data/ISPY2/case_000/
  V0.nii.gz
  V1.nii.gz
  V2.nii.gz
  V3.nii.gz
  V4.nii.gz
  mask.nii.gz
```

The loader also accepts `sequence.nii.gz` / `dce.nii.gz` as 4D input. Patient-level splits default to `train:val:test = 7:1:2`.

## Synthetic Debug Dataset

No real DUKE/ISPY2 data is required to check the full pipeline.

```bash
python main.py --command synthetic --config configs/rhdsnet_ispy2.yaml --output_dir data/synthetic_ispy2
python main.py --command train --config configs/rhdsnet_ispy2.yaml
python main.py --command test --config configs/rhdsnet_ispy2.yaml
```

Synthetic post-contrast phases contain gradual tumor enhancement with mostly stable background.

## Train RHDSNet

```bash
python main.py --command train --config configs/rhdsnet_ispy2.yaml
python main.py --command train --config configs/rhdsnet_duke.yaml
```

Defaults:

- Adam, learning rate `1e-4`
- batch size `1`
- crop size `96 x 96 x 48`
- diffusion steps `1000`
- linear beta schedule `1e-6 -> 1e-2`
- diffusion pretrain `200` epochs
- max epochs `1000`
- early stopping patience `50`
- five seeds: `2024..2028`
- best checkpoint by validation Dice
- mixed precision, DataParallel, and `torchrun` DistributedDataParallel through config flags

## Test RHDSNet

```bash
python main.py --command test --config configs/rhdsnet_ispy2.yaml --ckpt outputs/rhdsnet_ispy2/seed_2024/best.pth
```

Testing uses sliding-window inference. Outputs include CSV/JSON metrics, prediction masks, and generated post-contrast volumes when enabled.

## SOTA Proxy Comparison

```bash
python main.py --command sota --config configs/rhdsnet_ispy2.yaml
```

Outputs:

```text
Methods | Input | ISPY2 DSC | ISPY2 HD95 | ISPY2 Sen | DUKE DSC | DUKE HD95 | DUKE Sen | Params | FLOPs
```

Important: baseline modules under `models/baselines/` are **faithful proxy implementations for experimental comparison, not official implementations**. If official code is available, replace the corresponding module while keeping the same forward interface.

## Generate-then-Segment

```bash
python main.py --command generate_then_segment --config configs/rhdsnet_ispy2.yaml
```

This trains a full-sequence TSESNet-like oracle proxy, trains V0-to-DCE generators, feeds generated sequences into the same segmenter, and compares against RHDSNet.

Output:

```text
Methods | mean-SSIM | DSC | HD95 | Sen
```

## DB/LD Ablation

```bash
python main.py --command ablation_db_ld --config configs/ablation_db_ld.yaml
```

Compares:

- DDPM baseline: no Dynamic Blending, no Local Decoupling
- `+DB`
- `+LD`
- RHDSNet: DB + LD

## MTC Ablation

```bash
python main.py --command ablation_mtc --config configs/ablation_mtc.yaml
```

Output:

```text
hf1 | hf2 | hf3 | hf4 | DSC | HD95 | Sen
```

## Recursive Noise Robustness

```bash
python main.py --command robustness --config configs/robustness.yaml --ckpt outputs/rhdsnet_ispy2/seed_2024/best.pth
```

Tests no perturbation and Gaussian noise injected at `V1..V4`, then continues recursive generation.

## Statistical Test

```bash
python main.py --command stats --result_dir outputs/sota --dataset ISPY2
```

The script searches saved `patient_metrics.csv` files and reports:

- mean ± std
- 95% confidence interval of paired differences
- paired t-test p-value

## Visualization

```bash
python main.py --command visualize \
  --case_npz data/synthetic_ispy2/case_000.npz \
  --pred_dir outputs \
  --output_dir outputs/visualizations \
  --axis axial
```

Supported outputs:

- segmentation qualitative comparison with red overlays
- DCE-MRI generation comparison
- axial/sagittal/coronal slices
- optional yellow zoom box via the Python API
- simple marching-cubes OBJ export when `scikit-image` is installed

## Inference Rule

RHDSNet inference calls only:

```python
model(v0)
```

No real post-contrast phase is passed into RHDD or the segmentation network during inference. Generated volumes are used internally only to carry recurrent RHDD state; segmentation consumes RHDD encoder features through MTC, not raw generated phases.
