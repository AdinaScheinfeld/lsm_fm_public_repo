# Scripts — Segmentation

Scripts for finetuning LSM foundation model for binary segmentation.

---

## Contents

**Shared**
- [finetune_segmentation_utils.py](finetune_segmentation_utils.py) - Shared utilities, dataset, and base Lightning module used by both backbones.

**UNet**
- [finetune_segmentation_unet.py](finetune_segmentation_unet.py) - UNet finetuning and inference script for segmentation.
- [finetune_segmentation_unet_config.yaml](finetune_segmentation_unet_config.yaml) - Config file for UNet segmentation finetuning.
- [finetune_segmentation_unet_job.sh](finetune_segmentation_unet_job.sh) - Job script to run UNet finetuning and inference for segmentation.

**SwinUNETR**
- [finetune_segmentation_swinunetr.py](finetune_segmentation_swinunetr.py) - SwinUNETR finetuning and inference script for segmentation.
- [finetune_segmentation_swinunetr_config.yaml](finetune_segmentation_swinunetr_config.yaml) - Config file for SwinUNETR segmentation finetuning.
- [finetune_segmentation_swinunetr_job.sh](finetune_segmentation_swinunetr_job.sh) - Job script to run SwinUNETR finetuning and inference for segmentation.

---

## Quick Start

### Step 1 - Configure Training

Edit the config file for your chosen backbone. The key fields to update are the paths to the data directories, output directory, and pretrained checkpoint.

**For UNet:** edit `finetune_segmentation_unet_config.yaml`

**For SwinUNETR:** edit `finetune_segmentation_swinunetr_config.yaml`

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
cd 02-finetuning/scripts/segmentation
sbatch finetune_segmentation_unet_job.sh
```

**SwinUNETR:**
```bash
cd 02-finetuning/scripts/segmentation
sbatch finetune_segmentation_swinunetr_job.sh
```

Outputs written to `02-finetuning/output/`:
- Checkpoints in `output/checkpoints/`
- Binary predictions and probability maps for test images in `output/preds/`
- Per-file and mean Dice@0.5 for test images in `output/preds/metrics_test.csv`
- Logs in `output/logs/`
- Training progress, losses, and validation image samples are logged to WandB

---

## Advanced

### Finetune on Your Own Data

Point `train_dir`, `val_dir`, and `test_dir` in the config to your own data directories. Your data should be in NIfTI format with segmentation masks named with the additional suffix `_label`. For example: `amyloid_plaque_patch_000_vol000_ch0_label.nii.gz` is the label corresponding to `amyloid_plaque_patch_000_vol000_ch0.nii.gz`.

### Additional Configuration Options

| Parameter | Description |
|-----------|-------------|
| `channel_substr` | Which channel to use from the images (default: ALL) |
| `file_prefix` | Prefix to select only specific patch types for train/val/test (default: null) |
| `min_train` | Minimum number of patches required for training (default: 1) |
| `min_val` | Minimum number of patches required for validation (default: 1) |
| `lr` | Learning rate (default: 0.0003) |
| `freeze_encoder_epochs` | Number of epochs to freeze encoder before finetuning it (default: 10) |
| `encoder_lr_mult` | Learning rate multiplier for the encoder relative to the decoder head (default: 0.2) |
| `max_epochs` | Maximum number of training epochs (default: 600) |
| `early_stopping_patience` | Epochs with no val_loss improvement before stopping (default: 200) |
| `loss_name` | Loss function: `dicefocal` or `dicece` (default: dicefocal) |
| `init` | `pretrained` to load from checkpoint, `scratch` to train from random initialization |

The following architecture-specific parameters must match the values used during pretraining:

**UNet** — edit the `unet_*` fields in `finetune_segmentation_unet_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `unet_channels` | Feature channels at each encoder level (default: 32,64,128,256,512) |
| `unet_strides` | Downsampling strides for each level (default: 2,2,2,1) |
| `unet_num_res_units` | Number of residual units per level (default: 2) |
| `unet_norm` | Normalization type: `BATCH`, `INSTANCE`, or `LAYER` (default: BATCH) |

**SwinUNETR** — edit the `swinunetr_*` fields in `finetune_segmentation_swinunetr_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `swinunetr_img_size` | Spatial size of input patches (default: 96) |
| `swinunetr_feature_size` | Feature size controlling model capacity (default: 48) |

### Hardware

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
```