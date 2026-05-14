# Setup

Environment and model setup required before running pretraining or finetuning.

---

## Contents

| File | Description |
|---|---|
| [`environment.yaml`](environment.yaml) | Conda environment specification — installs all required Python packages |
| [`download_bert_model.py`](download_bert_model.py) | Downloads and saves `bert-base-uncased` locally *(image+text pretraining only)* |

---

## Step 1 — Create the Conda Environment

```bash
conda env create -f setup/environment.yaml
conda activate lsm-pretrain
```

This installs all required packages including:

| Package | Version | Purpose |
|---|---|---|
| `torch` | 2.6.0 | Deep learning framework |
| `pytorch-lightning` | 2.5.1 | Training loop and distributed training |
| `monai` | 1.4.0 | Medical image transforms and losses |
| `transformers` | 4.54.1 | BERT model for image+text pretraining |
| `nibabel` | 5.3.2 | NIfTI file loading |
| `wandb` | 0.19.11 | Experiment tracking and logging |
| `umap-learn` | 0.5.9 | Embedding visualization |
| `numpy` | 1.26.4 | Numerical computing |
| `scikit-learn` | 1.6.1 | Metrics and utilities |

Python version: **3.9** · CUDA: **12.4**

---

## Step 2 — Download BERT *(image+text pretraining only)*

If you plan to run image+text pretraining (`model.use_text: true`), download the BERT model before training:

```bash
python setup/download_bert_model.py
```

This saves `bert-base-uncased` to `setup/pretrained_models/bert-base-uncased/`. The pretraining config files already point to this path by default — no changes needed.

> **Skip this step** if running image-only pretraining or finetuning only.
