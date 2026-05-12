# Scripts

Scripts for finetuning LSM foundation model.

## Contents

**Shared**
- [segmentation/finetune_segmentation_utils.py](segmentation/finetune_segmentation_utils.py) - Shared utilities, dataset, and base Lightning module used by both backbones.

**UNet**
- [segmentation/finetune_segmentation_unet.py](segmentation/finetune_segmentation_unet.py) - UNet finetuning and inference script for segmentation.
- [segmentation/finetune_segmentation_unet_config.yaml](segmentation/finetune_segmentation_unet_config.yaml) - Config file for UNet segmentation finetuning.
- [segmentation/finetune_segmentation_unet_job.sh](segmentation/finetune_segmentation_unet_job.sh) - Job script to run UNet finetuning and inference for segmentation.

**SwinUNETR**
- [segmentation/finetune_segmentation_swinunetr.py](segmentation/finetune_segmentation_swinunetr.py) - SwinUNETR finetuning and inference script for segmentation.
- [segmentation/finetune_segmentation_swinunetr_config.yaml](segmentation/finetune_segmentation_swinunetr_config.yaml) - Config file for SwinUNETR segmentation finetuning.
- [segmentation/finetune_segmentation_swinunetr_job.sh](segmentation/finetune_segmentation_swinunetr_job.sh) - Job script to run SwinUNETR finetuning and inference for segmentation.