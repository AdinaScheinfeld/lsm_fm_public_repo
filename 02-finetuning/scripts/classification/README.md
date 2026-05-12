# Scripts — Classification

Scripts for finetuning LSM foundation model for multiclass patch classification.

---

## Contents

**Shared**
- [finetune_classification_utils.py](finetune_classification_utils.py) - Shared utilities, dataset, and base logic used by both backbones.

**UNet**
- [finetune_classification_unet.py](finetune_classification_unet.py) - UNet finetuning and inference script for classification.
- [finetune_classification_unet_config.yaml](finetune_classification_unet_config.yaml) - Config file for UNet classification finetuning.
- [finetune_classification_unet_job.sh](finetune_classification_unet_job.sh) - Job script to run UNet finetuning and inference for classification.

**SwinUNETR**
- [finetune_classification_swinunetr.py](finetune_classification_swinunetr.py) - SwinUNETR finetuning and inference script for classification.
- [finetune_classification_swinunetr_config.yaml](finetune_classification_swinunetr_config.yaml) - Config file for SwinUNETR classification finetuning.
- [finetune_classification_swinunetr_job.sh](finetune_classification_swinunetr_job.sh) - Job script to run SwinUNETR finetuning and inference for classification.

---

## Data Format

Classification patches should be NIfTI files (`.nii.gz`) in a flat directory. The class name is inferred from the filename prefix up to `_patch_`:

```
train/
    amyloid_plaque_patch_000_vol000_ch0.nii.gz   -> class: amyloid_plaque
    amyloid_plaque_patch_001_vol001_ch0.nii.gz   -> class: amyloid_plaque
    cell_nucleus_patch_002_vol007_ch0.nii.gz     -> class: cell_nucleus
    vessels_patch_014_vol013_ch0.nii.gz          -> class: vessels
```

Label files (ending in `_label.nii.gz`) are automatically skipped, so segmentation and classification patches can share the same directories.

---

## Quick Start

### Step 1 - Configure Training

Edit the config file for your chosen backbone. The key fields to update are the paths to the data directories, output directory, and pretrained checkpoint.

**For UNet:** edit `finetune_classification_unet_config.yaml`

**For SwinUNETR:** edit `finetune_classification_swinunetr_config.yaml`

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
cd 02-finetuning/scripts/classification
sbatch finetune_classification_unet_job.sh
```

**SwinUNETR:**
```bash
cd 02-finetuning/scripts/classification
sbatch finetune_classification_swinunetr_job.sh
```

Outputs written to `02-finetuning/output/`:
- Checkpoints in `output/checkpoints/`
- Per-patch predictions in `output/preds/preds.csv`
- Confusion matrix in `output/preds/confusion_matrix.csv`
- Per-class precision, recall, and F1 in `output/preds/per_class_metrics.csv`
- Logs in `output/logs/`
- Training progress and accuracy are logged to WandB

---

## Advanced

### Finetune on Your Own Data

Point `train_dir`, `val_dir`, and `test_dir` in the config to your own data directories. If you do not have separate val and test directories, comment out `val_dir` and `test_dir` and the data will be automatically split from `train_dir` using `train_percent` and `val_percent`.

Your patches should be named `<class_name>_patch_<anything>.nii.gz`. The class name is everything before the first `_patch_` in the filename.

### Additional Configuration Options

| Parameter | Description |
|-----------|-------------|
| `channel_substr` | Which channel to use from the images (default: ALL) |
| `train_percent` | Fraction of data to use for training when auto-splitting (default: 0.70) |
| `val_percent` | Fraction of data to use for validation when auto-splitting (default: 0.15) |
| `min_train` | Minimum patches required for training (default: 1) |
| `min_val` | Minimum patches required for validation (default: 1) |
| `min_test` | Minimum patches required for test (default: 1) |
| `lr` | Learning rate (default: 0.0001) |
| `freeze_encoder_epochs` | Number of epochs to freeze encoder before finetuning it (default: 5) |
| `linear_probe` | If `true`, freeze encoder for entire training and only train classification head (default: false) |
| `max_epochs` | Maximum number of training epochs (default: 200) |
| `early_stopping_patience` | Epochs with no val_accuracy improvement before stopping (default: 50) |
| `class_weighting` | Class imbalance strategy: `none`, `inverse_freq`, or `effective_num` (default: inverse_freq) |
| `init` | `pretrained` to load from checkpoint, `scratch` to train from random initialization |

The following architecture-specific parameters must match the values used during pretraining:

**UNet** — edit the `unet_*` fields in `finetune_classification_unet_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `unet_channels` | Feature channels at each encoder level (default: 32,64,128,256,512) |
| `unet_strides` | Downsampling strides for each level (default: 2,2,2,1) |
| `unet_num_res_units` | Number of residual units per level (default: 2) |
| `unet_norm` | Normalization type: `BATCH`, `INSTANCE`, or `LAYER` (default: BATCH) |

**SwinUNETR** — edit the `swinunetr_feature_size` field in `finetune_classification_swinunetr_config.yaml`:

| Parameter | Description |
|-----------|-------------|
| `swinunetr_feature_size` | Feature size controlling model capacity (default: 48) |

### Hardware

```bash
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
```