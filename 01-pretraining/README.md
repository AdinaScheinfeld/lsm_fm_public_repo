# Light Sheet Microscopy Foundation Model — Pretraining

This directory contains code for pretraining 3D foundation models on light sheet microscopy (LSM) image patches, with support for both image+text and image-only pretraining. Two backbone architectures are supported: UNet and SwinUNETR.

---

## Contents

- [sample_patches](sample_patches/) - Subfolder for sample NIfTI image files and JSON text prompts.
- [scripts](scripts/) - Code for pretraining LSM foundation model.

---

## Quick Start

### Requirements

- Linux (tested on Ubuntu; required for SLURM job submission)
- NVIDIA GPU(s) with CUDA support (tested on H100 96GB)
- Conda


### Pretrained Weights Available

If you do not want to run pretraining yourself, pretrained UNet and SwinUNETR checkpoints (both image+text and image-only) are available for download at [https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516). To use them, skip directly to `02-finetuning/` and follow the finetuning README.

### Step 1 - Clone the Repo

Clone this repo so you have your own working copy.

### Step 2 - Create the Conda Environment

```bash
conda env create -f setup/environment.yaml
conda activate lsm-pretrain
```

This installs all required Python packages including PyTorch, MONAI, PyTorch Lightning, HuggingFace Transformers, and WandB.

### Step 3 - Download the BERT Model (required for image+text mode only)

If you plan to run image+text pretraining, download and save the BERT model locally before training.

```bash
python setup/download_bert_model.py
```

This saves `bert-base-uncased` to `setup/pretrained_models/bert-base-uncased/`. The config files already point to this location by default.

### Step 4 - Download Training Data (optional)

The full pretraining dataset is available on Zenodo at [https://doi.org/10.5281/zenodo.20149070](https://doi.org/10.5281/zenodo.20149070). Five dataset archives are provided, each as a `.tar.gz` file:

| Archive | Source |
|---|---|
| `all_wu_brain_patches.tar.gz` | Wu brain atlas patches |
| `all_selma_patches_96.tar.gz` | SELMA dataset patches |
| `all_allen_human2_patches.tar.gz` | Allen Human Brain Atlas patches |
| `all_allen_developing_mouse_patches.tar.gz` | Allen Developing Mouse Brain patches |
| `all_allen_connection_projection_patches.tar.gz` | Allen Mouse Brain Connectivity Atlas patches |

Download and extract whichever datasets you want to train on:

```bash
tar -xzf all_wu_brain_patches.tar.gz
tar -xzf all_selma_patches_96.tar.gz
# repeat for any other archives
```

If you only want to train on the sample patches included in the repo, skip this step.

### Step 5 - Configure Training

Edit the config file for your chosen backbone to match your setup.

**For UNet:** edit `01-pretraining/scripts/pretrain_unet_config.yaml`

**For SwinUNETR:** edit `01-pretraining/scripts/pretrain_swinunetr_config.yaml`

**Data paths** — point each source to the directory where you extracted the downloaded data, and set each source to `True` or `False` to enable or disable it:

```yaml
data:
  roots:
    connection: /path/to/all_allen_connection_projection_patches
    dev_mouse: /path/to/all_allen_developing_mouse_patches
    human2: /path/to/all_allen_human2_patches
    selma: /path/to/all_selma_patches_96
    wu: /path/to/all_wu_brain_patches

  enable:
    connection: True
    dev_mouse: True
    human2: True
    selma: True
    wu: True
```

**Training mode** — set `use_text: true` for image+text pretraining, `false` for image-only:
```yaml
model:
  use_text: true
```

### Step 6 - Configure the Job Script

Edit the job script for your chosen backbone and update the following:

**For UNet:** edit `01-pretraining/scripts/pretrain_unet_job.sh`

**For SwinUNETR:** edit `01-pretraining/scripts/pretrain_swinunetr_job.sh`

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

### Step 7 - Run Pretraining

Submit the job for your chosen backbone:

**UNet:**
```bash
cd 01-pretraining/scripts
sbatch pretrain_unet_job.sh
```

**SwinUNETR:**
```bash
cd 01-pretraining/scripts
sbatch pretrain_swinunetr_job.sh
```

This will pretrain the backbone using the sample patches and text descriptions provided in `01-pretraining/sample_patches/`. Checkpoints and logs will be written to `01-pretraining/output/`. Training progress, losses, and reconstructed image samples are logged to WandB.

---

## Advanced

### Using the Full Training Datasets

The full pretraining datasets are available on Zenodo at [https://doi.org/10.5281/zenodo.20149070](https://doi.org/10.5281/zenodo.20149070) — see [Step 4](#step-4---download-training-data-optional) for download instructions. Once extracted, each dataset has the following directory structure, which the data loading code expects automatically:

| Source | Config key | Expected layout | Text prompt file |
|--------|------------|----------------|-----------------|
| Allen Connection Projection | `connection` | `<root>/<run>/c0\|c1\|c2/*.nii.gz` | `text_prompts_allen_connection.json` |
| Allen Developing Mouse | `dev_mouse` | `<root>/*.nii.gz` (flat) | `text_prompts_allen_dev_mouse.json` |
| Allen Human2 | `human2` | `<root>/<subject>/*.nii.gz` | `text_prompts_allen_human2.json` |
| Selma | `selma` | `<root>/<class>/*.nii.gz` | `text_prompts_selma.json` |
| Wu Brain | `wu` | `<root>/<volume>/input/*.nii.gz` | `text_prompts_wu.json` |

No code changes are needed — just point each `roots` entry in the config to the extracted directory and set the corresponding `enable` flag to `True`.

---

### Additional Configuration Options

The following parameters can be edited in the config file for your chosen backbone:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `max_epochs` | `training` | Total number of epochs to train for |
| `patience` | `training` | Use for early stopping (set to a high value to effectively disable) |
| `downsample.enabled` | `downsample` | If `true`, downsamples $96^3$ patches to `target_size`$^3$ before training |
| `global_batch_size` | `data` | Total batch size (automatically divided across GPUs) (default: 16) |
| `mask_ratio_warmup` | `model` | Mask ratio used during warmup epochs (default: 0.05) |
| `warmup_epochs` | `model` | Number of warmup epochs |
| `mask_ratio` | `model` | Fraction of voxels masked during training (default: 0.3) |
| `lr` | `model` | Learning rate (default: 0.0001) |
| `mask_patch_size` | `model` | Base mask size (default: 8) |
| `save_dirpath` | `model` | Directory in which to save model checkpoints and logs |
| `distill_weight` | `loss_weights` | Weight for student-teacher KL divergence loss |
| `reconstruction_weight` | `loss_weights` | Weight for masked voxel L1 reconstruction loss |
| `align_weight` | `loss_weights` | Weight for per-pair image-text cosine alignment loss |
| `clip_weight` | `loss_weights` | Weight for contrastive image-text CLIP loss |

The following architecture-specific parameters can also be configured:

**UNet** — edit the `unet:` block in `pretrain_config_unet.yaml`:

| Parameter | Description |
|-----------|-------------|
| `channels` | Number of feature channels at each encoder level (default: [32, 64, 128, 256, 512]) |
| `strides` | Downsampling strides for each level (default: [2, 2, 2, 1]) |
| `num_res_units` | Number of residual units per level (default: 2) |
| `norm` | Normalization type: `BATCH`, `INSTANCE`, or `LAYER` (default: BATCH) |

**SwinUNETR** — edit the `swinunetr:` block in `pretrain_swinunetr_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `feature_size` | Controls model capacity; larger values increase parameters (default: 48) |
| `use_checkpoint` | Enable gradient checkpointing to reduce memory usage (default: true) |

---

### Hardware

You may also need to update the GPU configurations to match your available hardware:

```yaml
# config file

dist:
  devices: 2          # number of GPUs per node
  num_nodes: 1        # number of nodes
  accumulate_grad_batches: 8
```

```bash
# job script

#SBATCH --gres=gpu:2              # GPU count
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2       # must match number of GPUs
```

---

### Resuming Training

To resume training from the last checkpoint:
```bash
sbatch pretrain_unet_job.sh --resume
```

To resume from a specific checkpoint:
```bash
srun python pretrain_unet.py --config pretrain_config_unet.yaml --ckpt_path /path/to/checkpoint.ckpt
```

The same `--resume` and `--ckpt_path` flags are available for the SwinUNETR scripts.

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel
- All patches should be the same spatial size (default: 96×96×96 voxels)
- The model uses instance normalization and is robust to different intensity ranges — intensity is normalized to [0, 1] via percentile scaling during loading
- With `use_text: false`, BERT is never loaded and CLIP/alignment losses are zero — the model trains purely on masked image reconstruction and teacher distillation
- Prompt JSON files for disabled sources do not need to exist — missing files are skipped automatically
- `unet_strides` in the UNet config must match between pretraining and finetuning configs — if you change the default strides during pretraining, use the same values when finetuning
