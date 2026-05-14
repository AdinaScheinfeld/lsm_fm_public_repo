# Helpers

Shared utility functions used across both pretraining and finetuning pipelines.

---

## Contents

| File | Used by | Description |
|---|---|---|
| [`all_datasets_transforms.py`](all_datasets_transforms.py) | Pretraining · Finetuning | Data loading, normalization, and augmentation transforms |
| [`visualization_functions.py`](visualization_functions.py) | Pretraining | WandB logging for images, embeddings, and UMAP plots |

---

## `all_datasets_transforms.py`

Provides MONAI-based transform pipelines shared across all scripts. Three sets of transforms are defined:

| Function | Used by | What it does |
|---|---|---|
| `get_load_transforms(target_size)` | Pretraining | Loads NIfTI, normalizes intensity to [0, 1] via percentile clipping, optionally downsamples to `target_size`³ |
| `get_train_transforms()` | Pretraining | Spatial augmentations (flip, rotate, affine) + intensity augmentations (noise, smooth, scale, shift) |
| `get_val_transforms()` | Pretraining | Identity transform — loading is handled separately |
| `get_finetune_train_transforms()` | Finetuning | Normalizes intensity, then applies spatial and intensity augmentations to image+label pairs jointly |
| `get_finetune_val_transforms()` | Finetuning | Normalizes intensity only — no augmentation |

---

## `visualization_functions.py`

Provides WandB logging utilities used during pretraining. Three functions are defined:

| Function | What it logs |
|---|---|
| `log_intensity_histogram` | Pixel intensity histogram for a given image tensor |
| `log_images_batches_to_wandb_table` | Side-by-side table of original, masked, and reconstructed image slices (mid-z) across validation batches |
| `log_embedding_umap_plot` | 2D UMAP projection of image and text embeddings, color-coded by biological category and modality |

---

## Notes

- Both files are imported via `sys.path.append('../../00-helpers/')` in the pretraining and finetuning scripts — no installation required
- Intensity normalization uses 1st–99th percentile clipping, making the model robust to outlier intensities across different microscopy datasets
