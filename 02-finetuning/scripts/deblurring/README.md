# ✨ Deblurring

Finetune a pretrained LSM foundation model for **3D image deblurring** of light sheet microscopy patches.

<div align="center">
    <img src="../../../media/deblurring_fig.png" width="400" alt="Deblurring results">
</div>

| | UNet | SwinUNETR |
|---|---|---|
| **Config** | `finetune_deblurring_unet_config.yaml` | `finetune_deblurring_swinunetr_config.yaml` |
| **Job script** | `finetune_deblurring_unet_job.sh` | `finetune_deblurring_swinunetr_job.sh` |
| **Submit** | `sbatch finetune_deblurring_unet_job.sh` | `sbatch finetune_deblurring_swinunetr_job.sh` |

---

## Contents

| File | Description |
|---|---|
| [`finetune_deblurring_utils.py`](finetune_deblurring_utils.py) | Shared utilities, dataset, base Lightning module, and evaluation logic used by both backbones |
| [`finetune_deblurring_unet.py`](finetune_deblurring_unet.py) | UNet finetuning and inference script |
| [`finetune_deblurring_unet_config.yaml`](finetune_deblurring_unet_config.yaml) | UNet config — set data paths, checkpoint, and hyperparameters |
| [`finetune_deblurring_unet_job.sh`](finetune_deblurring_unet_job.sh) | UNet SLURM job script |
| [`finetune_deblurring_swinunetr.py`](finetune_deblurring_swinunetr.py) | SwinUNETR finetuning and inference script |
| [`finetune_deblurring_swinunetr_config.yaml`](finetune_deblurring_swinunetr_config.yaml) | SwinUNETR config — set data paths, checkpoint, and hyperparameters |
| [`finetune_deblurring_swinunetr_job.sh`](finetune_deblurring_swinunetr_job.sh) | SwinUNETR SLURM job script |

---

## Data Format

Patches should be NIfTI files (`.nii.gz`) in a flat directory. Each sharp patch must have a corresponding blurred version with the `_blurred` suffix in the same directory:

```
train/
    vessels_patch_014_vol013_ch0.nii.gz           ← sharp (training target)
    vessels_patch_014_vol013_ch0_blurred.nii.gz   ← blurred (model input)
    cell_nucleus_patch_002_vol007_ch0.nii.gz
    cell_nucleus_patch_002_vol007_ch0_blurred.nii.gz
    ...
```

Label files (ending in `_label.nii.gz`) and blurred files are automatically skipped when scanning for sharp images, so segmentation and deblurring patches can share the same directories.

---

## Quick Start

### Step 1 — Configure

Edit the config for your chosen backbone and update the key fields:

```yaml
train_dir:       ../../sample_patches/train/
val_dir:         ../../sample_patches/val/
test_dir:        ../../sample_patches/test/
out_root:        ../../output/
pretrained_ckpt: ../../../01-pretraining/output/pretrained_unet_best.ckpt
                 # or pretrained_swinunetr_best.ckpt
```

Pretrained checkpoints are available on Zenodo: [https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516). Set `pretrained_ckpt` to your downloaded checkpoint, or set `init: scratch` to train from random initialization.

### Step 2 — Update the job script

```bash
#SBATCH --partition=your-gpu-partition   # your cluster's GPU partition
module load anaconda3                    # your cluster's anaconda module
export WANDB_API_KEY=your_key_here       # or run `wandb login` interactively
```

### Step 3 — Run

Submit from this directory:

```bash
cd 02-finetuning/scripts/deblurring
sbatch finetune_deblurring_unet_job.sh      # UNet
sbatch finetune_deblurring_swinunetr_job.sh # SwinUNETR
```

Outputs written to `02-finetuning/output/`:

| Output | Location |
|---|---|
| Checkpoints | `output/checkpoints/` |
| Deblurred NIfTI predictions | `output/preds/` |
| Per-file and mean PSNR + SSIM | `output/preds/metrics_test.csv` |
| Logs | `output/logs/` |
| Training curves + image samples | WandB |

---

## Advanced

<details>
<summary>Finetune on your own data</summary>

Point `train_dir`, `val_dir`, and `test_dir` in the config to your own directories. If you don't have separate val and test sets, omit `val_dir` and `test_dir` — the data will be split automatically from `train_dir` using `train_percent` and `val_percent`.

Each sharp patch must have a corresponding blurred file named `<stem>_blurred.nii.gz` in the same directory. Files without a blurred counterpart are skipped with a warning.

</details>

<details>
<summary>Training hyperparameters</summary>

| Parameter | Description |
|-----------|-------------|
| `channel_substr` | Channel filter: `ch0`, `ch1`, or `ALL` (default: ALL) |
| `train_percent` | Fraction of data for training when auto-splitting (default: 0.70) |
| `val_percent` | Fraction of data for validation when auto-splitting (default: 0.15) |
| `min_train` | Minimum pairs required for training (default: 1) |
| `min_val` | Minimum pairs required for validation (default: 1) |
| `min_test` | Minimum pairs required for test (default: 1) |
| `lr` | Decoder learning rate (default: 0.0001) |
| `encoder_lr_mult` | Encoder LR as a fraction of decoder LR (default: 0.05 SwinUNETR, 0.1 UNet) |
| `freeze_encoder_epochs` | Epochs to freeze encoder before finetuning it (default: 0) |
| `max_epochs` | Maximum training epochs (default: 500) |
| `early_stopping_patience` | Epochs without val_total_loss improvement before stopping (default: 50) |
| `edge_weight` | Weight for 3D gradient edge loss (default: 0.1) |
| `highfreq_weight` | Weight for high-frequency detail loss (default: 0.1) |
| `init` | `pretrained` or `scratch` (default: pretrained) |

</details>

<details>
<summary>Architecture parameters — must match pretraining</summary>

**UNet:**

| Parameter | Description |
|-----------|-------------|
| `unet_channels` | Feature channels per encoder level (default: 32,64,128,256,512) |
| `unet_strides` | Downsampling strides per level (default: 2,2,2,1) |
| `unet_num_res_units` | Residual units per level (default: 2) |
| `unet_norm` | Normalization: `BATCH`, `INSTANCE`, or `LAYER` (default: BATCH) |
| `freeze_bn_stats` | Freeze BatchNorm running stats during training (default: false) |

**SwinUNETR:**

| Parameter | Description |
|-----------|-------------|
| `swinunetr_feature_size` | Model capacity — must match pretraining (default: 48) |

</details>

<details>
<summary>Loss function</summary>

The model predicts a **residual** added to the blurred input and clamped to [0, 1]:

```
output = clamp(blurred_input + model(blurred_input), 0, 1)
```

Composite loss: **L1 + 0.2 × SSIM + `edge_weight` × edge + `highfreq_weight` × highfreq**

Both `edge_weight` and `highfreq_weight` are configurable in the YAML config.

</details>

<details>
<summary>Hardware settings</summary>

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
```

</details>