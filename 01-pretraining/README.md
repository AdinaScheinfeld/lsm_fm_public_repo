# Light Sheet Microscopy Foundation Model — Pretraining

This directory contains code for pretraining a 3D UNet-based foundation model on light sheet microscopy (LSM) image patches, with support for both image+text and image-only pretraining.

---

## Quick Start

### Requirements

- Linux (tested on Ubuntu; required for SLURM job submission)
- NVIDIA GPU(s) with CUDA support (tested on H100 96GB)
- Conda


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

This saves `bert-base-uncased` to `setup/pretrained_models/bert-base-uncased/`. The config file already points to this location by default.

### Step 4 - Configure Training

Edit `01-pretraining/configs/pretrain_config_unet.yaml` to match your setup. The key fields to update are:

**Training mode** — set `use_text: true` for image+text pretraining, `false` for image-only:
```yaml
model:
  use_text: true
```

### Step 5 - Configure the Job Script

Edit `01-pretraining/job_scripts/pretrain_unet.sh` and update the following:

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

### Step 6 - Run Pretraining

Submit the job:
```bash
cd 01-pretraining/job_scripts
sbatch pretrain_unet.sh
```

This will pretrain a UNet backbone using the sample patches and text descriptions provided in `01-pretraining/sample_patches/`. Checkpoints and logs will be written to `01-pretraining/output/`. Training progress, losses, and reconstructed image samples are logged to WandB.

---

## Advanced

### Additional Datasets

You can add additional NIfTI patch files (`.nii.gz`) according to the structure expected by each data source. When pretraining with image+text, each source also requires a corresponding JSON prompt file mapping stain/region labels to text descriptions. The supported sources and their expected directory layouts are:

| Source | Expected layout | Text Descriptions |
|--------|----------------| -------------------- |
| Allen Connection Projection | `<root>/<run>/c0\|c1\|c2/*.nii.gz` | `text_prompts_allen_connection.json` |
| Allen Developing Mouse | `<root>/*.nii.gz` (flat) | `text_prompts_allen_dev_mouse.json` |
| Allen Human2 | `<root>/<subject>/*.nii.gz` | `text_prompts_allen_human2.json` |
| Selma | `<root>/<class>/*.nii.gz` | `text_prompts_selma.json` |
| Wu Brain | `<root>/<volume>/input/*.nii.gz` | `text_prompts_wu.json` |


**Enable/disable sources** — When adding additional datasets, be sure to set each source to `True` or `False` accordingly in `01-pretraining/configs/pretrain_config_unet.yaml`:
```yaml
  enable:
    connection: False
    dev_mouse: False
    human2: False
    selma: False
    wu: True
```
---

### Additional Configuration Options

The following parameters can be edited in `01-pretraining/configs/pretrain_config_unet.yaml`:

| Parameter | Location | Description |
|-----------|----------|-------------|
| `max_epochs` | `training` | Total number of epochs to train for |
| `patience` | `training` | Use for early stopping (set to a high value to effectively disable)
| `downsample.enabled` | `downsample` | If `true`, downsamples $96^3$ patches to `target_size`$^3$ before training |
| `global_batch_size` | `data` | Total batch size (automatically divided across GPUs) (default: 16) |
| `mask_ratio_warmup` | `model` | Mask ratio used during warmup epochs (default: 0.05) |
| `warmup_epochs` | `model` | Number of warmup epochs (default: 20) |
| `mask_ratio` | `model` | Fraction of voxels masked during training (default: 0.3) |
| `lr` | `model` | Learning rate (default: 0.0001) |
| `mask_patch_size` | `model` | Base mask size (default: 8) |
| `save_dirpath` | `model` | Directory in which to save model checkpoints and logs |
| `distill_weight` | `loss_weights` | Weight for student-teacher KL divergence loss (default: 0.29) |
| `reconstruction_weight` | `loss_weights` | Weight for masked voxel L1 reconstruction loss (default: 0.20) |
| `align_weight` | `loss_weights` | Weight for per-pair image-text cosine alignment loss (default: 0.06) |
| `clip_weight` | `loss_weights` | Weight for contrastive image-text CLIP loss (default: 0.36) |

---

### Hardware

You may also need to update the GPU configurations to match your available hardware

```yaml
# /01-pretraining/scripts/pretrain_config_unet.yaml

dist:
  devices: 2          # number of GPUs per node
  num_nodes: 1        # number of nodes
  accumulate_grad_batches: 8
```



```bash
# /01-pretraining/scripts/pretrain_unet_job.sh

#SBATCH --gres=gpu:h100:2         # GPU type and count
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=2       # must match number of GPUs
```

---

### Resuming Training

To resume training from the last checkpoint:
```bash
sbatch pretrain_unet.sh --resume
```

To resume from a specific checkpoint:
```bash
srun python pretrain_unet.py --config ../configs/pretrain_config_unet.yaml --ckpt_path /path/to/checkpoint.ckpt
```

---

## Notes

- Patch files must be in NIfTI format (`.nii.gz`), 3D, single-channel
- All patches should be the same spatial size (default: 96×96×96 voxels)
- The model uses instance normalization and is robust to different intensity ranges — intensity is normalized to [0, 1] via percentile scaling during loading
- With `use_text: false`, BERT is never loaded and CLIP/alignment losses are zero — the model trains purely on masked image reconstruction and teacher distillation
- Prompt JSON files for disabled sources do not need to exist — missing files are skipped automatically
