# Pretraining Scripts

Scripts for pretraining the LSM foundation model. For full setup and usage instructions see [01-pretraining/README.md](../README.md).

---

## At a Glance

| Backbone | To run | Config | Job script |
|---|---|---|---|
| UNet | `python pretrain_unet.py --config pretrain_unet_config.yaml` | `pretrain_unet_config.yaml` | `pretrain_unet_job.sh` |
| SwinUNETR | `python pretrain_swinunetr.py --config pretrain_swinunetr_config.yaml` | `pretrain_swinunetr_config.yaml` | `pretrain_swinunetr_job.sh` |

Submit via SLURM from this directory:
```bash
cd 01-pretraining/scripts
sbatch pretrain_unet_job.sh        # UNet
sbatch pretrain_swinunetr_job.sh   # SwinUNETR
```

---

## File Reference

### Shared

| File | Description |
|---|---|
| [`data_module.py`](data_module.py) | LightningDataModule — discovers files from all enabled sources, applies per-source sampling, builds train/val splits, and returns dataloaders for both image+text and image-only modes |
| [`multisource_patch_image_text_dataset.py`](multisource_patch_image_text_dataset.py) | Dataset for image+text pretraining — loads NIfTI patches and extracts text prompts from filename and directory structure |
| [`multisource_patch_image_only_dataset.py`](multisource_patch_image_only_dataset.py) | Dataset for image-only pretraining — loads NIfTI patches without text |

### UNet

| File | Description |
|---|---|
| [`pretrain_unet_config.yaml`](pretrain_unet_config.yaml) | Config file — set data paths, enable/disable sources, choose image+text or image-only mode, and tune all training hyperparameters |
| [`pretrain_module_unet.py`](pretrain_module_unet.py) | PyTorch Lightning module — implements the UNet student-teacher pretraining loop with masked reconstruction, distillation, and optional CLIP loss |
| [`pretrain_unet.py`](pretrain_unet.py) | Main entry point — parses config, initialises the data module and model, and launches training |
| [`pretrain_unet_job.sh`](pretrain_unet_job.sh) | SLURM job script — update partition, conda module, and WandB key before submitting |

### SwinUNETR

| File | Description |
|---|---|
| [`pretrain_swinunetr_config.yaml`](pretrain_swinunetr_config.yaml) | Config file — set data paths, enable/disable sources, choose image+text or image-only mode, and tune all training hyperparameters |
| [`pretrain_module_swinunetr.py`](pretrain_module_swinunetr.py) | PyTorch Lightning module — implements the SwinUNETR student-teacher pretraining loop with masked reconstruction, distillation, and optional CLIP loss |
| [`pretrain_swinunetr.py`](pretrain_swinunetr.py) | Main entry point — parses config, initialises the data module and model, and launches training |
| [`pretrain_swinunetr_job.sh`](pretrain_swinunetr_job.sh) | SLURM job script — update partition, conda module, and WandB key before submitting |