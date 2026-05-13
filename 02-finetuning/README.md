# Light Sheet Microscopy Foundation Model — Finetuning

Finetune a pretrained LSM foundation model on your own data. Three downstream tasks are supported, each with **UNet** and **SwinUNETR** backbones.

| Task | Input | Output | Metric |
|---|---|---|---|
| ✂️ [**Segmentation**](scripts/segmentation/README.md) | Image + binary label mask | Predicted binary mask | Dice@0.5 |
| 🏷️ [**Classification**](scripts/classification/README.md) | Image patch | Predicted class label | Accuracy, macro F1 |
| ✨ [**Deblurring**](scripts/deblurring/README.md) | Blurred/sharp image pair | Deblurred NIfTI prediction | PSNR, SSIM |

---

## Contents

- [`sample_patches/`](sample_patches/) — Sample NIfTI patches, binary label masks, and blurred patches for smoke testing.
- [`scripts/segmentation/`](scripts/segmentation/) — Binary segmentation finetuning scripts.
- [`scripts/classification/`](scripts/classification/) — Multiclass patch classification finetuning scripts.
- [`scripts/deblurring/`](scripts/deblurring/) — Image deblurring finetuning scripts.

---

## Quick Start

### Requirements

- Linux (tested on Ubuntu; required for SLURM job submission)
- NVIDIA GPU(s) with CUDA support (tested on H100 96GB)
- Conda

---

### Step 1 — Install

> **Already ran pretraining?** Skip to [Step 3](#step-3--get-a-pretrained-checkpoint).

```bash
git clone https://github.com/AdinaScheinfeld/lsm_fm_public_repo.git
cd lsm_fm_public_repo
conda env create -f setup/environment.yaml
conda activate lsm-pretrain
```

---

### Step 2 — Get a pretrained checkpoint

Pretrained checkpoints for all four model variants are available on Zenodo:

**[https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516)**

| Checkpoint | Backbone | Mode |
|---|---|---|
| `pretrained_unet_image_text.ckpt` | UNet | Image + text |
| `pretrained_unet_image_only.ckpt` | UNet | Image only |
| `pretrained_swinunetr_image_text.ckpt` | SwinUNETR | Image + text |
| `pretrained_swinunetr_image_only.ckpt` | SwinUNETR | Image only |

Download the checkpoint for your chosen backbone and note the path — you will set `pretrained_ckpt` in the config to point to it. To finetune from random initialization instead, set `init: scratch` in the config.

---

### Step 3 — Choose a task and follow its README

Each task has its own config, job script, and README with full instructions:

- ✂️ **Segmentation** — [scripts/segmentation/README.md](scripts/segmentation/README.md)
- 🏷️ **Classification** — [scripts/classification/README.md](scripts/classification/README.md)
- ✨ **Deblurring** — [scripts/deblurring/README.md](scripts/deblurring/README.md)

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel, default 96×96×96 voxels
- Intensity is normalized to [0, 1] during loading — the model is robust to different intensity ranges
- Architecture parameters (`unet_strides`, `unet_norm`, `swinunetr_feature_size`, etc.) must match between pretraining and finetuning configs — mismatches will cause weights to fail to load