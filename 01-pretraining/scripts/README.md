# Scripts

Scripts for pretraining LSM foundation model.

## Contents

**Shared**
- [data_module.py](data_module.py) - LightningDataModule that handles image+text and image-only datasets.
- [multisource_patch_image_only_dataset.py](multisource_patch_image_only_dataset.py) - Dataset class for loading image patches without text captions.
- [multisource_patch_image_text_dataset.py](multisource_patch_image_text_dataset.py) - Dataset class for image+text pretraining.

**UNet**
- [pretrain_config_unet.yaml](pretrain_config_unet.yaml) - Config file for UNet pretraining with image+text or image-only.
- [pretrain_module_unet.py](pretrain_module_unet.py) - PyTorch Lightning module for pretraining a UNet backbone with image+text or image-only.
- [pretrain_unet.py](pretrain_unet.py) - Main training script for UNet image+text and image-only pretraining.
- [pretrain_unet_job.sh](pretrain_unet_job.sh) - Job script to run UNet pretraining.

**SwinUNETR**
- [pretrain_swinunetr_config.yaml](pretrain_swinunetr_config.yaml) - Config file for SwinUNETR pretraining with image+text or image-only.
- [pretrain_module_swinunetr.py](pretrain_module_swinunetr.py) - PyTorch Lightning module for pretraining a SwinUNETR backbone with image+text or image-only.
- [pretrain_swinunetr.py](pretrain_swinunetr.py) - Main training script for SwinUNETR image+text and image-only pretraining.
- [pretrain_swinunetr_job.sh](pretrain_swinunetr_job.sh) - Job script to run SwinUNETR pretraining.