# ✂️ Segmentation

Finetune a pretrained LSM foundation model for **binary segmentation** of 3D light sheet microscopy patches.

<img src="../../../media/segmentation_fig.png" width="700" alt="Segmentation results">

| | UNet | SwinUNETR |
|---|---|---|
| **Config** | `finetune_segmentation_unet_config.yaml` | `finetune_segmentation_swinunetr_config.yaml` |
| **Job script** | `finetune_segmentation_unet_job.sh` | `finetune_segmentation_swinunetr_job.sh` |
| **Submit** | `sbatch finetune_segmentation_unet_job.sh` | `sbatch finetune_segmentation_swinunetr_job.sh` |

---

## Contents

| File | Description |
|---|---|
| [`finetune_segmentation_utils.py`](finetune_segmentation_utils.py) | Shared utilities, dataset, and base Lightning module used by both backbones |
| [`finetune_segmentation_unet.py`](finetune_segmentation_unet.py) | UNet finetuning and inference script |
| [`finetune_segmentation_unet_config.yaml`](finetune_segmentation_unet_config.yaml) | UNet config — set data paths, checkpoint, and hyperparameters |
| [`finetune_segmentation_unet_job.sh`](finetune_segmentation_unet_job.sh) | UNet SLURM job script |
| [`finetune_segmentation_swinunetr.py`](finetune_segmentation_swinunetr.py) | SwinUNETR finetuning and inference script |
| [`finetune_segmentation_swinunetr_config.yaml`](finetune_segmentation_swinunetr_config.yaml) | SwinUNETR config — set data paths, checkpoint, and hyperparameters |
| [`finetune_segmentation_swinunetr_job.sh`](finetune_segmentation_swinunetr_job.sh) | SwinUNETR SLURM job script |

---

## Data Format

Patches should be NIfTI files (`.nii.gz`) in a flat directory. Each image patch must have a corresponding binary label mask with the `_label` suffix:

```
train/
    amyloid_plaque_patch_000_vol000_ch0.nii.gz        ← image
    amyloid_plaque_patch_000_vol000_ch0_label.nii.gz  ← binary mask
    cell_nucleus_patch_002_vol007_ch0.nii.gz
    cell_nucleus_patch_002_vol007_ch0_label.nii.gz
    ...
```

Blurred files (ending in `_blurred.nii.gz`) are automatically skipped, so segmentation and deblurring patches can share the same directories.

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
cd 02-finetuning/scripts/segmentation
sbatch finetune_segmentation_unet_job.sh      # UNet
sbatch finetune_segmentation_swinunetr_job.sh # SwinUNETR
```

Outputs written to `02-finetuning/output/`:

| Output | Location |
|---|---|
| Checkpoints | `output/checkpoints/` |
| Binary predictions + probability maps | `output/preds/` |
| Per-file and mean Dice@0.5 | `output/preds/metrics_test.csv` |
| Logs | `output/logs/` |
| Training curves + image samples | WandB |

---

## Advanced

<details>
<summary>Finetune on your own data</summary>

Point `train_dir`, `val_dir`, and `test_dir` in the config to your own directories. If you don't have separate val and test sets, omit `val_dir` and `test_dir` — the data will be split automatically from `train_dir` using `val_percent`.

Label masks must be named `<image_stem>_label.nii.gz` and placed in the same directory as the image.

</details>

<details>
<summary>Training hyperparameters</summary>

| Parameter | Description |
|-----------|-------------|
| `channel_substr` | Channel filter: `ch0`, `ch1`, or `ALL` (default: ALL) |
| `file_prefix` | Filter patches by filename prefix, e.g. `amyloid_plaque_patch` (default: null) |
| `min_train` | Minimum patches required for training (default: 1) |
| `min_val` | Minimum patches required for validation (default: 1) |
| `lr` | Decoder learning rate (default: 0.0003) |
| `encoder_lr_mult` | Encoder LR as a fraction of decoder LR (default: 0.2) |
| `freeze_encoder_epochs` | Epochs to freeze encoder before finetuning it (default: 10) |
| `max_epochs` | Maximum training epochs (default: 600) |
| `early_stopping_patience` | Epochs without val_loss improvement before stopping (default: 200) |
| `loss_name` | Loss function: `dicefocal` or `dicece` (default: dicefocal) |
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

**SwinUNETR:**

| Parameter | Description |
|-----------|-------------|
| `swinunetr_img_size` | Input patch spatial size (default: 96) |
| `swinunetr_feature_size` | Model capacity — must match pretraining (default: 48) |

</details>

<details>
<summary>Hardware settings</summary>

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
```

</details>