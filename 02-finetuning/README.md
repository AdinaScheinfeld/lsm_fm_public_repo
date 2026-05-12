# Light Sheet Microscopy Foundation Model — Finetuning

This directory contains code for finetuning and running inference using a light sheet microscopy (LSM) 3D foundation model for segmentation. Two backbone architectures are supported: UNet and SwinUNETR.

---

## Contents

- [sample_patches](sample_patches/) - Subfolder for sample NIfTI image files and binary label masks.
- [scripts](scripts/) - Code for finetuning LSM foundation model.

---

## Quick Start

### Requirements

- Linux (tested on Ubuntu; required for SLURM job submission)
- NVIDIA GPU(s) with CUDA support (tested on H100 96GB)
- Conda

### Important

If you already ran pretraining using the scripts provided in `01-pretraining/`, skip to [Step 4](#step-4---configure-training), otherwise, begin with [Step 1](#step-1---clone-the-repo).


### Step 1 - Clone the Repo

Clone this repo so you have your own working copy.

### Step 2 - Create the Conda Environment

```bash
conda env create -f setup/environment.yaml
conda activate lsm-pretrain
```

This installs all required Python packages including PyTorch, MONAI, PyTorch Lightning, HuggingFace Transformers, and WandB.

### Step 3 - Download Pretrained Checkpoint

Download either the image+text pretrained model or the image-only pretrained model from [here](https://drive.google.com/drive/folders/1-cXnWTcOkkug4ugBsgrTmUXWEJjD6f9E?usp=sharing). Both UNet and SwinUNETR checkpoints are available.

### Step 4 - Configure Training

Edit the config file for your chosen backbone to match your setup. The key fields to update are the paths to the training patches, validation patches, test patches, output directory, and the location where the pretrained checkpoint is saved.

**For UNet:** edit `02-finetuning/scripts/segmentation/finetune_segmentation_unet_config.yaml`

**For SwinUNETR:** edit `02-finetuning/scripts/segmentation/finetune_segmentation_swinunetr_config.yaml`

```yaml
train_dir: ../../sample_patches/train/
val_dir: ../../sample_patches/val/
test_dir: ../../sample_patches/test/
out_root: ../../output/
pretrained_ckpt: ../../../01-pretraining/output/pretrained_unet_best.ckpt  # or pretrained_swinunetr_best.ckpt
```

### Step 5 - Configure the Job Script

Edit the job script for your chosen backbone and update the following:

**For UNet:** edit `02-finetuning/scripts/segmentation/finetune_segmentation_unet_job.sh`

**For SwinUNETR:** edit `02-finetuning/scripts/segmentation/finetune_segmentation_swinunetr_job.sh`

**SLURM resource requests** — update to match your cluster:
```bash
#SBATCH --partition=your-gpu-partition   # your cluster's GPU partition name
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

### Step 6 - Run Finetuning

Submit the job for your chosen backbone:

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

This will finetune and run inference using the sample patches and label masks provided in `02-finetuning/sample_patches/`.

Outputs written to `02-finetuning/output/`:
- Checkpoints in `output/checkpoints/`
- Binary predictions and probability maps for test images in `output/preds/`
- Per-file and mean Dice@0.5 for test images in `output/preds/metrics_test.csv`
- Logs in `output/logs/`
- Training progress, losses, and validation image samples are logged to WandB

---

## Advanced

### Finetune on Your Own Data

You can finetune one of the pretrained models using your own data by pointing the `train_dir`, `val_dir`, and `test_dir` paths in the config file to the locations where your data is stored.

**Note:** Your data should be in NIfTI format with the filenames structured in such a way that the segmentation masks have the same name as the image patches with the additional suffix _\_label_. For example: _amyloid\_plaque\_patch\_000\_vol000\_ch0\_label.nii.gz_ is the label corresponding to the image patch _amyloid\_plaque\_patch\_000\_vol000\_ch0.nii.gz_.

### Additional Configuration Options

The following parameters can be edited in the config file for your chosen backbone:

| Parameter | Description |
|-----------|-------------|
| `channel_substr` | Which channel to use from the images (default: ALL) |
| `file_prefix` | Prefix to use only specific datatypes for train/val/test (default: null, to run without filtering) |
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

---

### Hardware

You may also need to update the GPU configurations to match your available hardware:

```bash
# job script

#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
```

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel
- All patches should be the same spatial size (default: 96×96×96 voxels)
- The model is robust to different intensity ranges — intensity is normalized to [0, 1] via percentile scaling during loading
- Architecture parameters (`unet_strides`, `unet_norm`, `swinunetr_feature_size`, etc.) must match between pretraining and finetuning configs — mismatches will result in weights failing to load