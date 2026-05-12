# Light Sheet Microscopy Foundation Model

This repository provides code for pretraining and finetuning 3D foundation models for light sheet microscopy (LSM) image analysis. Two backbone architectures are supported — UNet and SwinUNETR — with pretraining modes for both image+text (CLIP-style) and image-only objectives. Three downstream finetuning tasks are supported: binary segmentation, multiclass patch classification, and image deblurring.

Pretrained model weights are available at [https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516). If you do not want to run pretraining yourself, download a checkpoint and skip directly to [02-finetuning/](02-finetuning/).

A full description of the methods and results is available in our paper (coming soon on arXiv).

---

## Repository Structure

```
lsm_fm_public_repo/
├── 00-helpers/
│   ├── all_datasets_transforms.py          # Shared data augmentation and loading transforms
│   └── visualization_functions.py          # WandB logging and visualization utilities
├── 01-pretraining/
│   ├── sample_patches/                     # Sample NIfTI patches and text prompt JSON files
│   └── scripts/                            # Pretraining scripts, configs, and job scripts
│       ├── data_module.py
│       ├── multisource_patch_image_only_dataset.py
│       ├── multisource_patch_image_text_dataset.py
│       ├── pretrain_module_unet.py
│       ├── pretrain_unet.py
│       ├── pretrain_unet_config.yaml
│       ├── pretrain_unet_job.sh
│       ├── pretrain_module_swinunetr.py
│       ├── pretrain_swinunetr.py
│       ├── pretrain_swinunetr_config.yaml
│       └── pretrain_swinunetr_job.sh
├── 02-finetuning/
│   ├── sample_patches/                     # Sample NIfTI patches, label masks, and blurred patches
│   └── scripts/
│       ├── segmentation/                   # Binary segmentation finetuning scripts
│       ├── classification/                 # Multiclass patch classification finetuning scripts
│       └── deblurring/                     # Image deblurring finetuning scripts
└── setup/
    ├── download_bert_model.py              # Downloads BERT model for image+text pretraining
    └── environment.yaml                    # Conda environment specification
```

---

## Getting Started

### Requirements

- Linux (tested on Ubuntu; required for SLURM job submission)
- NVIDIA GPU(s) with CUDA support (tested on H100 96GB)
- Conda

### Installation

```bash
git clone https://github.com/AdinaScheinfeld/lsm_fm_public_repo.git
cd lsm_fm_public_repo
conda env create -f setup/environment.yaml
conda activate lsm-pretrain
```

---

## Pretrained Weights

Pretrained checkpoints for all four model variants are available on Zenodo:

**[https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516)**

| Checkpoint | Backbone | Pretraining mode |
|---|---|---|
| `pretrained_unet_image_text.ckpt` | UNet | Image + text (CLIP-style) |
| `pretrained_unet_image_only.ckpt` | UNet | Image only |
| `pretrained_swinunetr_image_text.ckpt` | SwinUNETR | Image + text (CLIP-style) |
| `pretrained_swinunetr_image_only.ckpt` | SwinUNETR | Image only |

---

## Pretraining

To pretrain your own model from scratch, see [01-pretraining/README.md](01-pretraining/README.md).

Pretraining supports:
- **UNet** and **SwinUNETR** backbones
- **Image+text** mode: contrastive CLIP-style objective pairing 3D image patches with biological text prompts, combined with masked image reconstruction and teacher distillation
- **Image-only** mode: masked image reconstruction and teacher distillation without text

---

## Finetuning

To finetune a pretrained model on a downstream task, see [02-finetuning/README.md](02-finetuning/README.md).

Three downstream tasks are supported:

| Task | Description | Output |
|---|---|---|
| **Segmentation** | Binary segmentation of structures in 3D LSM patches | Predicted binary masks + Dice@0.5 metrics |
| **Classification** | Multiclass classification of 3D patch identity | Per-patch predictions + accuracy, F1 metrics |
| **Deblurring** | Blind 3D image deblurring from blurred/sharp patch pairs | Deblurred NIfTI predictions + PSNR, SSIM metrics |

Each task supports both UNet and SwinUNETR backbones and can be initialized from a pretrained checkpoint or trained from random initialization.

---

## Notes

- All patch files should be in NIfTI format (`.nii.gz`), 3D, and single-channel
- Default patch size is 96×96×96 voxels
- Intensity is normalized to [0, 1] during loading — the model is robust to different intensity ranges
- Architecture parameters (`unet_strides`, `unet_norm`, `swinunetr_feature_size`, etc.) must match between pretraining and finetuning configs

