# A Multimodal 3D Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring

Repo for MICCAI 2026 submission

## Abstract

Light sheet fluorescence microscopy (LSM) enables high-resolution, three-dimensional (3D) imaging of biological specimens, providing rich volumetric data for studying cellular organization, pathology, and vascular networks. However, the size, dimensionality, and annotation burden of LSM data make supervised deep learning approaches costly and difficult to scale. Additionally, despite the abundance of unannotated LSM volumes, foundation models for this modality remain underexplored due to computational challenges and the complexity of volumetric representation learning. In this work, we introduce a 3D foundation model for LSM data, pretrained on a large curated collection of 3D images spanning multiple organisms, stains, and imaging protocols. We learn transferable volumetric representations by jointly optimizing for masked reconstruction and image-text alignment. The pretrained backbone drastically reduces the annotation burden, enabling efficient, few-shot adaptation for varied downstream tasks. We evaluate this approach on downstream segmentation, classification, and deblurring. Our results demonstrate consistent improvements over baselines, (1) when measured using standard evaluation metrics and (2) when rigorously assessed by domain experts. This highlights the potential of foundation model pretraining to reduce annotation requirements while improving performance across diverse LSM analysis tasks.

<!-- Complete text is available [here](here). -->

Pretrained model checkpoints are available to download from Google Drive: [https://drive.google.com/drive/folders/1-cXnWTcOkkug4ugBsgrTmUXWEJjD6f9E?usp=sharing](https://drive.google.com/drive/folders/1-cXnWTcOkkug4ugBsgrTmUXWEJjD6f9E?usp=sharing)

## Contents

- See [01-pretraining](01-pretraining/) for information on pretraining a light sheet microscopy foundation model.

- See [02-finetuning](02-finetuning/) for information on finetuning a light sheet microscopy foundation model. 


