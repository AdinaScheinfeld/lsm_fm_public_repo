# Finetuning Scripts

Three downstream tasks are supported, each with **UNet** and **SwinUNETR** backbones. Pick a task below and follow its README.

---

| Task | Backbone | Input | Output | Metric |
|---|---|---|---|---|
| ✂️ [**Segmentation**](segmentation/README.md) | UNet · SwinUNETR | Image + label mask | Binary mask | Dice@0.5 |
| 🏷️ [**Classification**](classification/README.md) | UNet · SwinUNETR | Image patch | Class label | Accuracy, F1 |
| ✨ [**Deblurring**](deblurring/README.md) | UNet · SwinUNETR | Blurred/sharp pair | Deblurred image | PSNR, SSIM |

---

> **New to finetuning?** Start with [Segmentation](segmentation/README.md) — it's the simplest task and a good way to verify your setup end-to-end.

> **No pretrained checkpoint yet?** Download one from Zenodo: [https://doi.org/10.5281/zenodo.20146516](https://doi.org/10.5281/zenodo.20146516)