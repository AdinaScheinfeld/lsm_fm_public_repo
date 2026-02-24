# Light Sheet Microscopy Foundation Model — Finetuning

This directory contains code for finetuning and running inference using a light sheet microscopy (LSM) 3D UNet-based foundation model to do segmentation.

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

Download either the image+text pretrained model or the image-only pretrained model from [here](here).

### Step 4 - Configure Training

Edit `02-finetuning/scripts/finetune_segmentation_unet_config_unet.yaml` to match your setup. The key fields to update are the paths to the training patches, validation patches, test patches, output directory, and the location where the pretrained checkpoint is saved. 

```yaml
train_dir: ../sample_patches/train/
val_dir: ../sample_patches/val/
test_dir: ../sample_patches/test/
out_root: ../output/
pretrained_ckpt: ../../01-pretraining/output/pretrained_unet_best.ckpt
```

### Step 5 - Configure the Job Script

Edit `02-finetuning/scripts/finetune_segmentation_unet_job.sh` and update the following:

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

Submit the job:
```bash
cd 02-finetuning/scripts
sbatch finetune_segmentation_unet_job.sh
```

This will finetune and run inference using a UNet backbone with the sample patches and label masks provided in `02-finetuning/sample_patches/`. 

Outputs:
- Checkpoints in `/output/checkpoints/`
- Binary predictions and probability maps for test images in `output/preds/`
- Per-file and mean Dice@0.5 for test images in `output/preds/metrics_test.csv`
- Logs in `output/logs/`
- Training progress, losses, and samples are logged to WandbB

---

## Advanced

### Finetune on Your Own Data

You can finetune one of the pretrained models using your own data by pointing the `train_dir`, `val_dir`, and `test_dir` paths in `finetune_segmentation_unet_config.yaml` to the locations where your data is stored. 

**Note:** Your data should be in NIfTI format with the filenames structured in such a way that the segmentation masks have the same name as the image patches with the additional suffix _\_label_. For example: _amyloid\_plaque\_patch\_000\_vol000\_ch0\_label.nii.gz_ is the label corresponding the image patch _amyloid\_plaque\_patch\_000\_vol000\_ch0.nii.gz_. 

### Additional Configuration Options

The following parameters can be edited in `02-finetuning/scripts/finetune_segmentation_unet_config.yaml`:

| Parameter | Description |
|-----------| -------------|
| `channel_substr` | Which channel to use from the images (default: ALL) |
| `file_prefix` | Prefix to use only specific datatypes for train/val/test (default: null, to run without filtering) |
| `min_train` | Minimum number of patches required for training (default: 1) |
| `min_val` | Minimum number of patches required for validation (default: 1) |
| `lr` | Learning rate (default: 0.0003) |
| `freeze_encoder_epochs` | Number of epochs to freeze encoder before finetuning it (default: 10) |

---

### Hardware

You may also need to update the GPU configurations to match your available hardware

```yaml
# /02-finetuning/scripts/finetune_segmentation_unet_config_unet.yaml

devices: 2          # number of GPUs per node
num_nodes: 1        # number of nodes
accumulate_grad_batches: 8
```

```bash
# /02-finetuning/scripts/finetune_segmentation_unet_job.sh

#SBATCH --gres=gpu                 # GPU type and count
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1        # must match number of GPUs
```

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel
- All patches should be the same spatial size (default: 96×96×96 voxels)
- The model uses instance normalization and is robust to different intensity ranges — intensity is normalized to [0, 1] via percentile scaling during loading

