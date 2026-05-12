# finetune_classification_utils.py - Shared utilities, dataset, and base logic for classification finetuning

# --- Setup ---

# imports
import csv
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import os
from pathlib import Path
import random
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
import pytorch_lightning as pl


# -------------------------
# Utils
# -------------------------

# function to set all random seeds for reproducibility
def _seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# function to seed each dataloader worker differently (avoids identical augmentations across workers)
def _seed_worker(_):
    info = get_worker_info()
    if info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + info.id)
        random.seed(base_seed + info.id)


# helper to format seconds as H:MM:SS
def _format_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}s"


# -------------------------
# Data Discovery and Splitting
# -------------------------

# function to discover samples from a flat directory of NIfTI patch files
# expects files named <class_name>_patch_<rest>.nii.gz in a flat directory (no subdirectories needed)
# the class name is extracted from the filename prefix up to the first occurrence of "_patch_"
# label files (ending in _label.nii.gz) are automatically skipped
# optionally filters by channel substring (e.g. "ch0", "ch1", or "ALL" for no filtering)
def discover_samples(root_dir: Path, channel_substr="ALL"):

    # normalize channel filter
    substrings = None
    s = str(channel_substr).strip()
    if s and s.upper() != "ALL":
        substrings = [t.strip().lower() for t in s.split(",") if t.strip()]

    # collect all nifti files, skipping label files
    all_files = []
    for p in sorted(root_dir.glob("*.nii*")):
        lower = p.name.lower()
        if lower.endswith("_label.nii") or lower.endswith("_label.nii.gz"):
            continue  # skip segmentation label files
        if lower.endswith("_blurred.nii") or lower.endswith("_blurred.nii.gz"):
            continue
        if substrings is not None and not any(tok in lower for tok in substrings):
            continue  # skip if channel filter not matched
        all_files.append(p)

    # extract class name from filename prefix up to "_patch_"
    # e.g. "amyloid_plaque_patch_000_vol000_ch0.nii.gz" -> "amyloid_plaque"
    class_name_set = set()
    for p in all_files:
        stem = p.name.split(".nii")[0]  # strip extension(s)
        if "_patch_" in stem:
            class_name_set.add(stem.split("_patch_")[0])
        else:
            # fallback: use full stem as class name if "_patch_" not found
            class_name_set.add(stem)

    class_names = sorted(class_name_set)
    name_to_idx = {name: idx for idx, name in enumerate(class_names)}

    print(f"[INFO] discover_samples: root_dir={root_dir} channel_substr={channel_substr} -> {len(class_names)} classes: {class_names}", flush=True)

    # assign each file to its class
    samples = []
    for p in all_files:
        stem = p.name.split(".nii")[0]
        if "_patch_" in stem:
            class_name = stem.split("_patch_")[0]
        else:
            class_name = stem
        if class_name not in name_to_idx:
            print(f"[WARN] Could not assign class for file {p.name}, skipping.", flush=True)
            continue
        samples.append({"path": p, "label_idx": name_to_idx[class_name], "label_name": class_name})

    print(f"[INFO] Found {len(samples)} total patches across {len(class_names)} classes.", flush=True)
    return samples, class_names


# function to split samples into train, val, and test sets by percentage
def split_train_val_test(samples, train_percent=0.7, val_percent=0.15, seed=100, min_train=1, min_val=1, min_test=1):

    n = len(samples)
    if n < (min_train + min_val + min_test):
        raise ValueError(f"Not enough samples ({n}) for min_train={min_train} + min_val={min_val} + min_test={min_test}.")

    rng = random.Random(seed)
    arr = list(samples)
    rng.shuffle(arr)

    n_train = max(min_train, int(round(n * train_percent)))
    n_val = max(min_val, int(round(n * val_percent)))
    n_test = max(min_test, n - n_train - n_val)

    # clamp to available samples
    n_train = min(n_train, n - min_val - min_test)
    n_val = min(n_val, n - n_train - min_test)
    n_test = n - n_train - n_val

    if n_train < min_train or n_val < min_val or n_test < min_test:
        raise ValueError(f"Split failed: train={n_train} val={n_val} test={n_test} with minimums {min_train}/{min_val}/{min_test}.")

    print(f"[INFO] Split {n} samples -> train={n_train} val={n_val} test={n_test}", flush=True)
    return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]


# function to compute class weights for loss weighting to handle class imbalance
# scheme: "none" (equal weights), "inverse_freq" (1/count), or "effective_num" (from Cui et al. 2019)
def compute_class_weights(train_samples, num_classes, scheme="inverse_freq", beta=0.9999):

    counts = [0] * num_classes
    for rec in train_samples:
        counts[rec["label_idx"]] += 1

    if scheme == "none":
        return [1.0] * num_classes
    elif scheme == "inverse_freq":
        return [1.0 / max(1, c) for c in counts]
    elif scheme == "effective_num":
        ws = []
        for c in counts:
            if c <= 0:
                ws.append(0.0)
            else:
                ws.append((1.0 - beta) / (1.0 - beta ** c))
        return ws

    raise ValueError(f"Invalid class weight scheme: {scheme}. Must be 'none', 'inverse_freq', or 'effective_num'.")


# -------------------------
# Dataset
# -------------------------

# function to safely load a NIfTI file and return float32 array
def _safe_load_nii(path):
    import nibabel as nib
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


# function to min-max normalize a patch to [0, 1]
def _min_max_normalize(patch, eps=1e-6):
    arr_min = float(np.min(patch))
    arr_max = float(np.max(patch))
    if arr_max - arr_min < eps:
        return np.zeros_like(patch, dtype=np.float32)
    return ((patch - arr_min) / (arr_max - arr_min)).astype(np.float32)


# function to apply random 3D flip augmentations
def _maybe_augment(x: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])  # W flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-2])  # H flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-3])  # D flip
    return x


# dataset to load NIfTI patches for classification
class PatchClassificationDataset(Dataset):

    def __init__(self, samples, augment=False, channel_substr="ALL"):
        self.augment = augment

        # apply channel filter
        substrings = None
        s = str(channel_substr).strip()
        if s and s.upper() != "ALL":
            substrings = [sub.strip().lower() for sub in s.split(",") if sub.strip()]

        if substrings is not None:
            self.samples = [rec for rec in samples if any(tok in rec["path"].name.lower() for tok in substrings)]
        else:
            self.samples = list(samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        path = rec["path"]
        y = int(rec["label_idx"])

        vol = _safe_load_nii(path)
        vol = _min_max_normalize(vol)
        x = torch.from_numpy(vol).unsqueeze(0)  # add channel dim: (1, D, H, W)

        if self.augment:
            x = _maybe_augment(x)

        return {
            "image": x,                          # tensor (1, D, H, W)
            "label": torch.tensor(y, dtype=torch.long),
            "filename": str(path.name),
            "label_name": rec["label_name"],
        }


# -------------------------
# Metrics
# -------------------------

# dataclass to hold run outputs
@dataclass
class RunOutputs:
    best_ckpt_path: str
    preds_csv: Path


# function to compute per-class and overall metrics from a confusion matrix
# cm: 2D numpy array with rows=true labels, cols=predicted labels
def metrics_from_counts(cm):

    K = cm.shape[0]
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    acc = tp.sum() / max(1, cm.sum())
    macro_f1 = f1.mean() if K > 0 else 0.0

    return acc, macro_f1, prec, rec, f1, support


# -------------------------
# Shared Evaluation Logic
# -------------------------

# function to run inference on test set and write predictions and metrics to CSV
def run_evaluation(model, test_loader, class_names, device, output_dir: Path, tag: str, wandb_logger=None):

    output_dir.mkdir(parents=True, exist_ok=True)
    K = len(class_names)
    cm = np.zeros((K, K), dtype=np.int64)
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            yi = int(y.item())
            pi = int(pred.item())
            cm[yi, pi] += 1
            rows.append({
                "filename": batch["filename"][0],
                "true_label": class_names[yi],
                "predicted_label": class_names[pi],
                "pred_idx": pi,
                "true_idx": yi,
                "prob_pred": float(probs[0, pi].item()),
            })

    acc, macro_f1, prec, rec, f1, support = metrics_from_counts(cm)
    print(f"[INFO] Test Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}", flush=True)

    if wandb_logger:
        wandb_logger.experiment.summary[f"{tag}_test_accuracy"] = acc
        wandb_logger.experiment.summary[f"{tag}_test_macro_f1"] = macro_f1

    # save predictions CSV
    preds_csv = output_dir / "preds.csv"
    with open(preds_csv, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"[INFO] Saved predictions to: {preds_csv}", flush=True)

    # save confusion matrix CSV
    cm_csv = output_dir / "confusion_matrix.csv"
    with open(cm_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, cname in enumerate(class_names):
            w.writerow([cname] + list(map(int, cm[i].tolist())))
    print(f"[INFO] Saved confusion matrix to: {cm_csv}", flush=True)

    # save per-class metrics CSV
    metrics_csv = output_dir / "per_class_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "support", "precision", "recall", "f1_score"])
        for i, cname in enumerate(class_names):
            w.writerow([cname, int(support[i]), f"{prec[i]:.6f}", f"{rec[i]:.6f}", f"{f1[i]:.6f}"])
        w.writerow(["__OVERALL__", int(cm.sum()), f"ACC={acc:.6f}", f"MACRO_F1={macro_f1:.6f}", ""])
    print(f"[INFO] Saved per-class metrics to: {metrics_csv}", flush=True)

    return preds_csv


# -------------------------
# Shared Config Loader
# -------------------------

def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# -------------------------
# Shared Dataloader Builder
# -------------------------

# function to build train, val, and test dataloaders from config and discovered samples
def build_dataloaders(cfg, seed):

    train_dir = Path(cfg["train_dir"])
    val_dir   = cfg.get("val_dir", None)
    test_dir  = cfg.get("test_dir", None)
    channel_substr = cfg.get("channel_substr", "ALL")

    # always discover training samples first to establish the canonical class name -> index mapping
    train_samples, class_names = discover_samples(train_dir, channel_substr=channel_substr)

    # if separate val_dir and test_dir are provided, discover them using the same class map
    if val_dir is not None and test_dir is not None:
        val_samples,  val_classes  = discover_samples(Path(val_dir),  channel_substr=channel_substr)
        test_samples, test_classes = discover_samples(Path(test_dir), channel_substr=channel_substr)

        # remap val and test samples to the training class indices for consistency
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        for rec in val_samples + test_samples:
            if rec["label_name"] in name_to_idx:
                rec["label_idx"] = name_to_idx[rec["label_name"]]
            else:
                print(f"[WARN] Class '{rec['label_name']}' in val/test not found in training classes {class_names} — skipping.", flush=True)
        val_samples  = [r for r in val_samples  if r["label_name"] in name_to_idx]
        test_samples = [r for r in test_samples if r["label_name"] in name_to_idx]

        print(f"[INFO] Using separate val_dir ({len(val_samples)} samples) and test_dir ({len(test_samples)} samples).", flush=True)

    # otherwise auto-split train_dir by percentage
    else:
        train_percent = float(cfg.get("train_percent", 0.70))
        val_percent   = float(cfg.get("val_percent",   0.15))
        train_samples, val_samples, test_samples = split_train_val_test(
            train_samples,
            train_percent=train_percent,
            val_percent=val_percent,
            seed=seed,
            min_train=int(cfg.get("min_train", 1)),
            min_val=int(cfg.get("min_val", 1)),
            min_test=int(cfg.get("min_test", 1)),
        )

    print(f"[INFO] Train={len(train_samples)} Val={len(val_samples)} Test={len(test_samples)} Classes={len(class_names)}", flush=True)

    # compute class weights from training set only (to avoid data leakage)
    class_weighting = cfg.get("class_weighting", "inverse_freq")
    beta = float(cfg.get("beta", 0.9999))
    class_weights = compute_class_weights(train_samples, len(class_names), scheme=class_weighting, beta=beta)
    print(f"[INFO] Class weights ({class_weighting}): {class_weights}", flush=True)

    # build datasets and dataloaders
    num_workers = min(int(cfg.get("num_workers", 4)), os.cpu_count() or 4)
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
    )

    train_loader = DataLoader(PatchClassificationDataset(train_samples, augment=True,  channel_substr=channel_substr),
                              batch_size=int(cfg.get("batch_size", 8)), shuffle=True,  **loader_kw)
    val_loader   = DataLoader(PatchClassificationDataset(val_samples,   augment=False, channel_substr=channel_substr),
                              batch_size=1, shuffle=False, **loader_kw)
    test_loader  = DataLoader(PatchClassificationDataset(test_samples,  augment=False, channel_substr=channel_substr),
                              batch_size=1, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, class_names, class_weights

    