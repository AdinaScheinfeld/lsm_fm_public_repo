# Scripts — Deblurring

Scripts for finetuning LSM foundation model for 3D image deblurring.

---

## Contents

**Shared**
- [finetune_deblurring_utils.py](finetune_deblurring_utils.py) - Shared utilities, dataset, base Lightning module, and evaluation logic used by both backbones.

**UNet**
- [finetune_deblurring_unet.py](finetune_deblurring_unet.py) - UNet finetuning and inference script for deblurring.
- [finetune_deblurring_unet_config.yaml](finetune_deblurring_unet_config.yaml) - Config file for UNet deblurring finetuning.
- [finetune_deblurring_unet_job.sh](finetune_deblurring_unet_job.sh) - Job script to run UNet finetuning and inference for deblurring.

**SwinUNETR**
- [finetune_deblurring_swinunetr.py](finetune_deblurring_swinunetr.py) - SwinUNETR finetuning and inference script for deblurring.
- [finetune_deblurring_swinunetr_config.yaml](finetune_deblurring_swinunetr_config.yaml) - Config file for SwinUNETR deblurring finetuning.
- [finetune_deblurring_swinunetr_job.sh](finetune_deblurring_swinunetr_job.sh) - Job script to run SwinUNETR finetuning and inference for deblurring.

---

## Data Format

Deblurring patches should be NIfTI files (`.nii.gz`) in a flat directory. Each sharp patch must have a corresponding blurred version in the same directory, named with the `_blurred` suffix inserted before the extension:

```
train/
    vessels_patch_014_vol013_ch0.nii.gz           (sharp — used as training target)
    vessels_patch_014_vol013_ch0_blurred.nii.gz   (blurred — used as model input)
    cell_nucleus_patch_002_vol007_ch0.nii.gz
    cell_nucleus_patch_002_vol007_ch0_blurred.nii.gz
    ...
```

Label files (ending in `_label.nii.gz`) and blurred files are automatically skipped when scanning for sharp images, so segmentation and deblurring patches can share the same directories.

---

## Quick Start

### Step 1 - Configure Training

Edit the config file for your chosen backbone. The key fields to update are the paths to the data directories, output directory, and pretrained checkpoint.

Pretrained UNet and SwinUNETR checkpoints (both image+text and image-only) are available at [https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516). Download the checkpoint for your chosen backbone and set `pretrained_ckpt` in the config to point to it. To train from random initialization instead, set `init: scratch`.

**For UNet:** edit `finetune_deblurring_unet_config.yaml`

**For SwinUNETR:** edit `finetune_deblurring_swinunetr_config.yaml`

```yaml
train_dir: ../../sample_patches/train/
val_dir: ../../sample_patches/val/
test_dir: ../../sample_patches/test/
out_root: ../../output/
pretrained_ckpt: ../../../01-pretraining/output/pretrained_unet_best.ckpt  # or pretrained_swinunetr_best.ckpt
```

### Step 2 - Configure the Job Script

Edit the job script for your chosen backbone and update the following:

**SLURM resource requests** — update to match your cluster:
```bash
#SBATCH --partition=your-gpu-partition
```

**Conda module** — update to match your cluster's anaconda module:
```bash
module load anaconda3        # update version string to match your cluster
source activate lsm-pretrain
```

**WandB** — set your API key before submitting:
```bash
export WANDB_API_KEY=your_key_here
```
Or run `wandb login` interactively before submitting the job.

### Step 3 - Run Finetuning

Submit the job from this directory:

**UNet:**
```bash
cd 02-finetuning/scripts/deblurring
sbatch finetune_deblurring_unet_job.sh
```

**SwinUNETR:**
```bash
cd 02-finetuning/scripts/deblurring
sbatch finetune_deblurring_swinunetr_job.sh
```

Outputs written to `02-finetuning/output/`:
- Checkpoints in `output/checkpoints/`
- Deblurred NIfTI predictions for test images in `output/preds/`
- Per-file and mean PSNR and SSIM for test images in `output/preds/metrics_test.csv`
- Logs in `output/logs/`
- Training progress, losses, and validation image samples are logged to WandB

---

## Advanced

### Finetune on Your Own Data

Point `train_dir`, `val_dir`, and `test_dir` in the config to your own data directories. If you do not have separate val and test directories, comment out `val_dir` and `test_dir` and the data will be automatically split from `train_dir` using `train_percent` and `val_percent`.

Each sharp patch must have a corresponding blurred file named `<stem>_blurred.nii.gz` in the same directory. If a blurred counterpart is not found for a sharp file, that file is skipped with a warning.

### Additional Configuration Options

| Parameter | Description |
|-----------|-------------|
| `channel_substr` | Which channel to use from the images (default: ALL) |
| `train_percent` | Fraction of data to use for training when auto-splitting (default: 0.70) |
| `val_percent` | Fraction of data to use for validation when auto-splitting (default: 0.15) |
| `min_train` | Minimum pairs required for training (default: 1) |
| `min_val` | Minimum pairs required for validation (default: 1) |
| `min_test` | Minimum pairs required for test (default: 1) |
| `lr` | Learning rate for the decoder (default: 0.0001) |
| `encoder_lr_mult` | Learning rate multiplier for the encoder (default: 0.05 for SwinUNETR, 0.1 for UNet) |
| `freeze_encoder_epochs` | Number of epochs to freeze the encoder before finetuning it (default: 0) |
| `max_epochs` | Maximum number of training epochs (default: 500) |
| `early_stopping_patience` | Epochs with no val_total_loss improvement before stopping (default: 50) |
| `edge_weight` | Weight for 3D gradient edge loss (default: 0.1) |
| `highfreq_weight` | Weight for high-frequency detail loss (default: 0.1) |
| `init` | `pretrained` to load from checkpoint, `scratch` to train from random initialization |

The following architecture-specific parameters must match the values used during pretraining:

**UNet** — edit the `unet_*` fields in `finetune_deblurring_unet_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `unet_channels` | Feature channels at each encoder level (default: 32,64,128,256,512) |
| `unet_strides` | Downsampling strides for each level (default: 2,2,2,1) |
| `unet_num_res_units` | Number of residual units per level (default: 2) |
| `unet_norm` | Normalization type: `BATCH`, `INSTANCE`, or `LAYER` (default: BATCH) |
| `freeze_bn_stats` | If `true`, freeze BatchNorm running stats during training (default: false) |

**SwinUNETR** — edit the `swinunetr_feature_size` field in `finetune_deblurring_swinunetr_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `swinunetr_feature_size` | Feature size controlling model capacity (default: 48) |

### Loss Function

The deblurring model uses a composite loss: **L1 + 0.2×SSIM + `edge_weight`×edge + `highfreq_weight`×highfreq**. The model predicts a residual that is added to the blurred input and clamped to [0, 1] to produce the deblurred output. Both `edge_weight` and `highfreq_weight` are configurable in the YAML config.

### Hardware

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
```

