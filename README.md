# A Multimodal Foundation Model for Light Sheet Fluorescence Microscopy Enables Few-Shot Segmentation, Classification, and Deblurring

Repo for MICCAI 2026 submission

## Description

Light sheet fluorescence microscopy (LSFM) enables high-resolution, three-dimensional imaging of biological specimens, providing rich volumetric data for studying cellular organization, pathology, and vascular networks. However, the size, dimensionality, and annotation burden of LSFM data make supervised deep learning approaches costly and difficult to scale. Additionally, despite the abundance of unannotated LSFM volumes, foundation models for this modality remain underexplored due to computational challenges and the complexity of volumetric representation learning. In this work, we introduce a 3D foundation model for LSFM data that is pretrained on a large curated collection of 3D image patches spanning multiple organisms, stains, and imaging protocols. We learn transferable volumetric representations by jointly optimizing for masked reconstruction and image-text alignment. The pretrained backbone drastically reduces the annotation burden, enabling efficient, few-shot adaptation for varied downstream tasks. We evaluate this approach on downstream segmentation, classification, and deblurring tasks. Our results demonstrate consistent improvements over baselines, highlighting the potential of foundation model pretraining to substantially reduce annotation requirements while improving performance across diverse light sheet microscopy tasks.

Complete text is available [here](here).
