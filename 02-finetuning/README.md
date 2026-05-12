# Light Sheet Microscopy Foundation Model — Finetuning

This directory contains code for finetuning and running inference using a light sheet microscopy (LSM) 3D foundation model. Three downstream tasks are supported: **segmentation**, **classification**, and **deblurring**. For each task, two backbone architectures are available: UNet and SwinUNETR.

---

## Contents

- [sample_patches](sample_patches/) - Subfolder for sample NIfTI image files, binary label masks, and blurred patches.
- [scripts](scripts/) - Code for finetuning LSM foundation model.
  - [scripts/segmentation](scripts/segmentation/) - Binary segmentation finetuning scripts.
  - [scripts/classification](scripts/classification/) - Multiclass patch classification finetuning scripts.
  - [scripts/deblurring](scripts/deblurring/) - Image deblurring finetuning scripts.

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

Choose your task and backbone, then follow the relevant README:

- **Segmentation** — see [scripts/segmentation/README.md](scripts/segmentation/README.md)
- **Classification** — see [scripts/classification/README.md](scripts/classification/README.md)
- **Deblurring** — see [scripts/deblurring/README.md](scripts/deblurring/README.md)

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel
- All patches should be the same spatial size (default: 96×96×96 voxels)
- The model is robust to different intensity ranges — intensity is normalized to [0, 1] during loading
- Architecture parameters (`unet_strides`, `unet_norm`, `swinunetr_feature_size`, etc.) must match between pretraining and finetuning configs — mismatches will result in weights failing to load