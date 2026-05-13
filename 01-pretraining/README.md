# Light Sheet Microscopy Foundation Model — Pretraining

Pretrain a 3D LSM foundation model on your own data using either a **UNet** or **SwinUNETR** backbone, with **image+text** (CLIP-style) or **image-only** objectives.

> **Don't want to pretrain?** Pretrained checkpoints (UNet and SwinUNETR, image+text and image-only) are available on Zenodo: [https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516). Download a checkpoint and jump straight to [`02-finetuning/`](../02-finetuning/README.md).

---

## Contents

- [sample_patches](sample_patches/) - Sample NIfTI image files and JSON text prompts.
- [scripts](scripts/) - Pretraining scripts, configs, and job scripts.

---

## Pretraining Modes

| | Image + text | Image only |
|---|---|---|
| **Objective** | CLIP contrastive loss + masked reconstruction + teacher distillation | Masked reconstruction + teacher distillation |
| **Requires** | BERT model + text prompt JSON files | Nothing extra |
| **Best for** | Data with known biological labels | General morphology, unlabelled data |
| **Config field** | `model.use_text: true` | `model.use_text: false` |

---

## Quick Start

### Requirements

- Linux (tested on Ubuntu; required for SLURM job submission)
- NVIDIA GPU(s) with CUDA support (tested on H100 96GB)
- Conda

---

### Step 1 — Install

```bash
git clone https://github.com/AdinaScheinfeld/lsm_fm_public_repo.git
cd lsm_fm_public_repo
conda env create -f setup/environment.yaml
conda activate lsm-pretrain
```

---

### Step 2 — Download BERT *(image+text mode only)*

Skip this step if running image-only pretraining (`model.use_text: false`).

```bash
python setup/download_bert_model.py
```

This saves `bert-base-uncased` to `setup/pretrained_models/bert-base-uncased/`. The config files already point to this path by default.

---

### Step 3 — Get training data

**Option A — Use the sample patches** included in the repo. No download needed — the configs already point to `01-pretraining/sample_patches/` by default. Good for a quick smoke test.

**Option B — Download the full datasets** from Zenodo: [https://doi.org/10.5281/zenodo.20149070](https://doi.org/10.5281/zenodo.20149070)

<details>
<summary>Show dataset details and directory layouts</summary>

Five archives are available, each as a `.tar.gz` file:

| Archive | Source |
|---|---|
| `all_wu_brain_patches.tar.gz` | Wu brain atlas patches |
| `all_selma_patches_96.tar.gz` | SELMA dataset patches |
| `all_allen_human2_patches.tar.gz` | Allen Human Brain Atlas patches |
| `all_allen_developing_mouse_patches.tar.gz` | Allen Developing Mouse Brain patches |
| `all_allen_connection_projection_patches.tar.gz` | Allen Mouse Brain Connectivity Atlas patches |

```bash
tar -xzf all_wu_brain_patches.tar.gz
# repeat for any other archives
```

Once extracted, each dataset has the following structure — the data loading code handles these automatically, no code changes needed:

| Source | Config key | Directory layout | Text prompt file |
|--------|------------|-----------------|-----------------|
| Allen Connection Projection | `connection` | `<root>/<run>/c0\|c1\|c2/*.nii.gz` | `text_prompts_allen_connection.json` |
| Allen Developing Mouse | `dev_mouse` | `<root>/*.nii.gz` (flat) | `text_prompts_allen_dev_mouse.json` |
| Allen Human2 | `human2` | `<root>/<subject>/*.nii.gz` | `text_prompts_allen_human2.json` |
| Selma | `selma` | `<root>/<class>/*.nii.gz` | `text_prompts_selma.json` |
| Wu Brain | `wu` | `<root>/<volume>/input/*.nii.gz` | `text_prompts_wu.json` |

</details>

---

### Step 4 — Configure

Edit the config for your chosen backbone:

| Backbone | Config file | Job script |
|---|---|---|
| UNet | `scripts/pretrain_unet_config.yaml` | `scripts/pretrain_unet_job.sh` |
| SwinUNETR | `scripts/pretrain_swinunetr_config.yaml` | `scripts/pretrain_swinunetr_job.sh` |

**If using the full datasets**, update the `roots` paths and `enable` flags:

```yaml
data:
  roots:
    connection: /path/to/all_allen_connection_projection_patches
    dev_mouse:  /path/to/all_allen_developing_mouse_patches
    human2:     /path/to/all_allen_human2_patches
    selma:      /path/to/all_selma_patches_96
    wu:         /path/to/all_wu_brain_patches
  enable:
    connection: true
    dev_mouse:  true
    human2:     true
    selma:      true
    wu:         true
```

**Set the pretraining mode:**

```yaml
model:
  use_text: true   # image+text | set to false for image-only
```

**Update the job script** to match your cluster:

```bash
#SBATCH --partition=your-gpu-partition   # GPU partition name
module load anaconda3                    # update to your cluster's module name
export WANDB_API_KEY=your_key_here       # or run `wandb login` interactively
```

---

### Step 5 — Run

Submit from the `scripts/` directory:

```bash
cd 01-pretraining/scripts

# UNet
sbatch pretrain_unet_job.sh

# SwinUNETR
sbatch pretrain_swinunetr_job.sh
```

Outputs are written to `01-pretraining/output/`:
- Best and last checkpoints in `output/`
- Periodic checkpoints in `output/periodic_checkpoints/`
- Logs in `output/logs/`
- Training progress, losses, and reconstructed image samples logged to WandB

---

## Resuming Training

```bash
# Resume from the last checkpoint automatically
sbatch pretrain_unet_job.sh --resume

# Resume from a specific checkpoint
srun python pretrain_unet.py --config pretrain_unet_config.yaml --ckpt_path /path/to/checkpoint.ckpt
```

The same flags work for the SwinUNETR scripts.

---

## Advanced Configuration

<details>
<summary>Training hyperparameters</summary>

| Parameter | Config section | Description |
|-----------|---------------|-------------|
| `max_epochs` | `training` | Total training epochs |
| `patience` | `training` | Early stopping patience (set high to disable) |
| `global_batch_size` | `data` | Total batch size across all GPUs (default: 16) |
| `mask_ratio` | `model` | Fraction of voxels masked per patch (default: 0.3) |
| `mask_ratio_warmup` | `model` | Mask ratio during warmup epochs (default: 0.05) |
| `warmup_epochs` | `model` | Number of warmup epochs |
| `lr` | `model` | Learning rate (default: 0.0001) |
| `mask_patch_size` | `model` | Base mask patch size (default: 8) |
| `save_dirpath` | `model` | Directory to save checkpoints |
| `downsample.enabled` | `data` | Downsample 96³ patches to `target_size`³ before training |

</details>

<details>
<summary>Loss weights</summary>

| Parameter | Description |
|-----------|-------------|
| `distill_weight` | Student-teacher KL divergence loss (default: 0.50) |
| `reconstruction_weight` | Masked voxel L1 reconstruction loss (default: 0.30) |
| `align_weight` | Per-pair image-text cosine alignment loss (default: 0.05) |
| `clip_weight` | Contrastive image-text CLIP loss (default: 0.30) |

`align_weight` and `clip_weight` are ignored when `use_text: false`.

</details>

<details>
<summary>Architecture parameters</summary>

These must match between pretraining and finetuning configs.

**UNet** — `unet:` block in `pretrain_unet_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `channels` | Feature channels per encoder level (default: [32, 64, 128, 256, 512]) |
| `strides` | Downsampling strides per level (default: [2, 2, 2, 1]) |
| `num_res_units` | Residual units per level (default: 2) |
| `norm` | Normalization: `BATCH`, `INSTANCE`, or `LAYER` (default: BATCH) |

**SwinUNETR** — `swinunetr:` block in `pretrain_swinunetr_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `feature_size` | Model capacity — larger = more parameters (default: 48) |
| `use_checkpoint` | Gradient checkpointing to reduce memory (default: true) |

</details>

<details>
<summary>Multi-GPU and hardware settings</summary>

Update both the config file and job script to match your hardware:

```yaml
# config file
dist:
  devices: 2                 # GPUs per node
  num_nodes: 1               # number of nodes
  accumulate_grad_batches: 8 # effective batch size multiplier
```

```bash
# job script
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2  # must equal number of GPUs
```

</details>

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel, default 96×96×96 voxels
- Intensity is normalized to [0, 1] via percentile scaling — the model is robust to different intensity ranges
- With `use_text: false`, BERT is never loaded and CLIP/alignment losses are disabled automatically
- Prompt JSON files for disabled sources do not need to exist — missing files are skipped
- `unet_strides` and `unet_norm` must match between pretraining and finetuning configs