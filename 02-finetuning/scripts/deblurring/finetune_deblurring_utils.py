# finetune_deblurring_utils.py - Shared utilities, dataset, base Lightning module, and evaluation logic for deblurring finetuning

# --- Setup ---

# imports
import csv
from dataclasses import dataclass
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import random
import sys
import time

import wandb

from monai.losses import SSIMLoss

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, get_worker_info

torch.set_float32_matmul_precision("medium")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


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

# dataclass to hold a sharp/blurred image pair
@dataclass
class DeblurPair:
    sharp: Path
    blurred: Path


# function to discover sharp/blurred NIfTI pairs in a flat directory
# expects files named <stem>.nii.gz (sharp) and <stem>_blurred.nii.gz (blurred) in the same directory
# label files (ending in _label.nii.gz) are automatically skipped
# optionally filters by channel substring (e.g. "ch0", "ch1", or "ALL" for no filtering)
def discover_pairs(data_dir: Path, channel_substr="ALL"):

    # normalize channel filter
    substrings = None
    s = str(channel_substr).strip()
    if s and s.upper() != "ALL":
        substrings = [t.strip().lower() for t in s.split(",") if t.strip()]

    pairs = []
    for p in sorted(data_dir.glob("*.nii*")):
        lower = p.name.lower()

        # skip label files and blurred files when scanning for sharp images
        if lower.endswith("_label.nii") or lower.endswith("_label.nii.gz"):
            continue
        if lower.endswith("_blurred.nii") or lower.endswith("_blurred.nii.gz"):
            continue

        # apply channel filter
        if substrings is not None and not any(tok in lower for tok in substrings):
            continue

        # derive the corresponding blurred filename by inserting "_blurred" before the extension
        suffix = "".join(p.suffixes)  # e.g. ".nii.gz"
        stem = p.name[: -len(suffix)]  # e.g. "vessels_patch_017_vol006_ch1"
        blurred = p.with_name(f"{stem}_blurred{suffix}")

        if not blurred.exists():
            print(f"[WARN] No blurred counterpart found for {p.name} — skipping.", flush=True)
            continue

        pairs.append(DeblurPair(sharp=p, blurred=blurred))

    print(f"[INFO] discover_pairs: {data_dir} -> {len(pairs)} sharp/blurred pairs.", flush=True)
    return pairs


# function to split pairs into train, val, and test sets by percentage
def split_train_val_test(pairs, train_percent=0.7, val_percent=0.15, seed=100, min_train=1, min_val=1, min_test=1):

    n = len(pairs)
    if n < (min_train + min_val + min_test):
        raise ValueError(f"Not enough pairs ({n}) for min_train={min_train} + min_val={min_val} + min_test={min_test}.")

    rng = random.Random(seed)
    arr = list(pairs)
    rng.shuffle(arr)

    n_train = max(min_train, int(round(n * train_percent)))
    n_val   = max(min_val,   int(round(n * val_percent)))
    n_train = min(n_train, n - min_val - min_test)
    n_val   = min(n_val,   n - n_train - min_test)
    n_test  = n - n_train - n_val

    if n_train < min_train or n_val < min_val or n_test < min_test:
        raise ValueError(f"Split failed: train={n_train} val={n_val} test={n_test} with minimums {min_train}/{min_val}/{min_test}.")

    print(f"[INFO] Split {n} pairs -> train={n_train} val={n_val} test={n_test}", flush=True)
    return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:]


# -------------------------
# Dataset
# -------------------------

# function to load a NIfTI file and return float32 array
def _load_nifti(path):
    nifti = nib.load(str(path))
    return nifti.get_fdata(dtype=np.float32), nifti.affine, nifti.header


# function to ensure volume is channel-first (C, D, H, W)
def _ensure_channel_first(arr):
    if arr.ndim == 3:
        return arr[None, ...]
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return np.transpose(arr, (3, 0, 1, 2))
    raise ValueError(f"Unsupported array shape: {arr.shape}")


# function to normalize both sharp and blurred images to [0, 1]
# percentiles are computed from the blurred image so normalization is consistent at inference time
def _normalize_pair(sharp, blurred, p_min=1.0, p_max=99.0):
    v = blurred.reshape(-1)
    vmin, vmax = np.percentile(v, [p_min, p_max])
    if vmax > vmin:
        blurred_norm = np.clip((blurred - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
        sharp_norm   = np.clip((sharp   - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    else:
        blurred_norm = np.zeros_like(blurred, dtype=np.float32)
        sharp_norm   = np.zeros_like(sharp,   dtype=np.float32)
    return sharp_norm, blurred_norm


# dataset to load sharp/blurred NIfTI pairs for deblurring
class NiftiDeblurDataset(Dataset):

    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        sharp_arr,   _, _ = _load_nifti(pair.sharp)
        blurred_arr, _, _ = _load_nifti(pair.blurred)

        sharp_vol   = _ensure_channel_first(sharp_arr)
        blurred_vol = _ensure_channel_first(blurred_arr)

        # normalize using percentiles from blurred image
        sharp_vol, blurred_vol = _normalize_pair(sharp_vol, blurred_vol)

        return {
            "input_vol":  torch.from_numpy(blurred_vol.copy()),  # (1, D, H, W) blurred input
            "target_vol": torch.from_numpy(sharp_vol.copy()),    # (1, D, H, W) sharp target
            "filename":   pair.sharp.name,
        }


# -------------------------
# Metrics
# -------------------------

# function to compute PSNR between predicted and target numpy arrays
def _calculate_psnr(pred, target, eps=1e-8):
    mse = float(np.mean((pred - target) ** 2)) + eps
    return 10.0 * np.log10(1.0 / mse)


# function to compute SSIM between predicted and target tensors using MONAI SSIMLoss
@torch.no_grad()
def _calculate_ssim(pred_t: torch.Tensor, target_t: torch.Tensor) -> float:
    if pred_t.ndim == 3:
        pred_t   = pred_t[None, None, ...]
    if target_t.ndim == 3:
        target_t = target_t[None, None, ...]
    ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0)
    return float(1.0 - ssim_loss_fn(pred_t.float(), target_t.float()).detach().cpu().item())


# function to save a deblurred volume as a NIfTI file, copying affine/header from the sharp reference
def _save_nifti(vol, ref_path: Path, out_path: Path):
    v = vol.detach().cpu().numpy()
    if v.ndim == 4:
        v = v[0]
    v = np.clip(v.astype(np.float32), 0.0, 1.0)
    ref = nib.load(str(ref_path))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v, affine=ref.affine, header=ref.header), str(out_path))


# -------------------------
# Base Lightning Module
# -------------------------

# base class for deblurring finetuning — shared loss computation, training/val steps, and optimizer
# subclasses must implement: __init__ (build self.model), configure_optimizers
class DeblurBaseModule(pl.LightningModule):

    def _init_shared(self, edge_weight: float, highfreq_weight: float):
        """Call this at the end of __init__ in each subclass to initialize shared loss state."""
        self.l1_loss        = nn.L1Loss(reduction="mean")
        self.ssim_loss_fn   = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.edge_weight    = float(edge_weight)
        self.highfreq_weight = float(highfreq_weight)
        self.val_examples_table = None

    def forward(self, x):
        return self.model(x)

    # *** Loss Helpers ***

    # 3D gradient for edge loss
    @staticmethod
    def _grad3d(x):
        dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        return dz, dy, dx

    # 3D high-pass filter (subtract local mean) to encourage matching fine details
    @staticmethod
    def _highpass3d(x, kernel_size=3):
        B, C, D, H, W = x.shape
        weight = x.new_ones((C, 1, kernel_size, kernel_size, kernel_size)) / float(kernel_size ** 3)
        smoothed = F.conv3d(x, weight, bias=None, stride=1, padding=kernel_size // 2, groups=C)
        return x - smoothed

    # compute all losses and PSNR from blurred input and sharp target
    def _compute_losses(self, blurred, sharp):
        residual = self(blurred)
        pred = torch.clamp(blurred + residual, 0.0, 1.0)

        loss_l1   = self.l1_loss(pred, sharp)
        loss_ssim = self.ssim_loss_fn(pred, sharp)  # SSIMLoss returns 1 - SSIM

        dz_p, dy_p, dx_p = self._grad3d(pred)
        dz_t, dy_t, dx_t = self._grad3d(sharp)
        loss_edge = (
            (dz_p.abs() - dz_t.abs()).abs().mean() +
            (dy_p.abs() - dy_t.abs()).abs().mean() +
            (dx_p.abs() - dx_t.abs()).abs().mean()
        )

        loss_hf = self.l1_loss(self._highpass3d(pred), self._highpass3d(sharp))

        total = loss_l1 + 0.2 * loss_ssim + self.edge_weight * loss_edge + self.highfreq_weight * loss_hf

        mse  = torch.mean((pred - sharp) ** 2) + 1e-8
        psnr = 10.0 * torch.log10(1.0 / mse)

        return pred, total, loss_l1, loss_ssim, loss_edge, loss_hf, psnr

    # *** Lightning Hooks ***

    def training_step(self, batch, batch_idx):
        blurred = batch["input_vol"].to(self.device)
        sharp   = batch["target_vol"].to(self.device)
        pred, total, l1, ssim_l, edge, hf, psnr = self._compute_losses(blurred, sharp)

        self.log("train_l1_loss",      l1,           on_step=False, on_epoch=True,  prog_bar=False)
        self.log("train_edge_loss",    edge,         on_step=False, on_epoch=True,  prog_bar=False)
        self.log("train_highfreq_loss",hf,           on_step=False, on_epoch=True,  prog_bar=False)
        self.log("train_total_loss",   total,        on_step=True,  on_epoch=True,  prog_bar=True)
        self.log("train_ssim",         1.0 - ssim_l, on_step=True,  on_epoch=True,  prog_bar=False)
        self.log("train_psnr",         psnr,         on_step=True,  on_epoch=True,  prog_bar=True)
        return total

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        blurred = batch["input_vol"].to(self.device)
        sharp   = batch["target_vol"].to(self.device)
        pred, total, l1, ssim_l, edge, hf, psnr = self._compute_losses(blurred, sharp)

        # log up to 5 sample images to WandB table per validation epoch
        if batch_idx < 5 and isinstance(self.logger, WandbLogger) and hasattr(self.logger, "experiment"):
            if self.val_examples_table is None:
                self.val_examples_table = wandb.Table(
                    columns=["Filename", "Blurred (mid-z)", "Deblurred (mid-z)", "Sharp (mid-z)"]
                )
            fname = batch["filename"][0] if "filename" in batch else f"val_{batch_idx}"
            mid = blurred.shape[2] // 2
            self.val_examples_table.add_data(
                fname,
                wandb.Image(blurred[0, 0, mid].detach().float().cpu().numpy(), caption="blurred"),
                wandb.Image(pred[0, 0, mid].detach().float().cpu().numpy(),    caption="deblurred"),
                wandb.Image(sharp[0, 0, mid].detach().float().cpu().numpy(),   caption="sharp"),
            )

        self.log("val_l1_loss",       l1,           on_step=False, on_epoch=True,  prog_bar=False)
        self.log("val_edge_loss",     edge,         on_step=False, on_epoch=True,  prog_bar=False)
        self.log("val_highfreq_loss", hf,           on_step=False, on_epoch=True,  prog_bar=False)
        self.log("val_total_loss",    total,        on_step=True,  on_epoch=True,  prog_bar=True)
        self.log("val_ssim",          1.0 - ssim_l, on_step=True,  on_epoch=True,  prog_bar=True)
        self.log("val_psnr",          psnr,         on_step=True,  on_epoch=True,  prog_bar=True)
        return {"val_total_loss": total}

    def on_validation_epoch_end(self):
        if self.val_examples_table is not None and isinstance(self.logger, WandbLogger) and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({f"val_examples_epoch_{self.current_epoch}": self.val_examples_table})
        self.val_examples_table = None


# -------------------------
# Shared Evaluation Logic
# -------------------------

# function to run inference on test set and write deblurred predictions and metrics to CSV
def run_evaluation(model, test_loader, test_pairs, preds_dir: Path, device, wandb_logger=None):

    preds_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            blurred = batch["input_vol"].to(device)
            sharp   = batch["target_vol"].to(device)
            fname   = Path(batch["filename"][0])

            residual = model(blurred)
            pred = torch.clamp(blurred + residual, 0.0, 1.0)

            # save deblurred prediction as NIfTI
            suffix   = "".join(fname.suffixes)
            stem     = fname.name[: -len(suffix)]
            out_path = preds_dir / f"{stem}_deblur_pred{suffix}"

            # find the original sharp file to copy affine/header from
            sharp_path = next((p.sharp for p in test_pairs if p.sharp.name == fname.name), None)
            if sharp_path is not None:
                _save_nifti(pred[0, 0], sharp_path, out_path)
            else:
                _save_nifti(pred[0, 0], fname, out_path)

            psnr = _calculate_psnr(pred[0, 0].detach().cpu().numpy(), sharp[0, 0].detach().cpu().numpy())
            ssim = _calculate_ssim(pred[0, 0], sharp[0, 0])

            rows.append({
                "filename":  fname.name,
                "psnr":      f"{psnr:.4f}",
                "ssim":      f"{ssim:.4f}",
                "pred_path": str(out_path),
            })
            print(f"[INFO] {fname.name}  PSNR={psnr:.4f}  SSIM={ssim:.4f}", flush=True)

    if rows:
        mean_psnr = float(np.mean([float(r["psnr"]) for r in rows]))
        mean_ssim = float(np.mean([float(r["ssim"]) for r in rows]))
        rows.append({"filename": "MEAN", "psnr": f"{mean_psnr:.4f}", "ssim": f"{mean_ssim:.4f}", "pred_path": ""})
        print(f"[INFO] Mean test PSNR={mean_psnr:.4f}  SSIM={mean_ssim:.4f}", flush=True)
        if wandb_logger:
            wandb_logger.experiment.summary["test_mean_psnr"] = mean_psnr
            wandb_logger.experiment.summary["test_mean_ssim"] = mean_ssim

    metrics_csv = preds_dir / "metrics_test.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "pred_path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Metrics saved to: {metrics_csv}", flush=True)

    return metrics_csv


# -------------------------
# Shared Config Loader and Dataloader Builder
# -------------------------

def load_config(path):
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


# function to build train, val, and test dataloaders from config and discovered pairs
def build_dataloaders(cfg, seed):

    train_dir = Path(cfg["train_dir"])
    val_dir   = cfg.get("val_dir", None)
    test_dir  = cfg.get("test_dir", None)
    channel_substr = cfg.get("channel_substr", "ALL")

    # always discover training pairs first
    train_pairs = discover_pairs(train_dir, channel_substr=channel_substr)

    # if separate val_dir and test_dir are provided, use them directly
    if val_dir is not None and test_dir is not None:
        val_pairs  = discover_pairs(Path(val_dir),  channel_substr=channel_substr)
        test_pairs = discover_pairs(Path(test_dir), channel_substr=channel_substr)
        print(f"[INFO] Using separate val_dir ({len(val_pairs)} pairs) and test_dir ({len(test_pairs)} pairs).", flush=True)

    # otherwise auto-split train_dir by percentage
    else:
        train_percent = float(cfg.get("train_percent", 0.70))
        val_percent   = float(cfg.get("val_percent",   0.15))
        train_pairs, val_pairs, test_pairs = split_train_val_test(
            train_pairs,
            train_percent=train_percent,
            val_percent=val_percent,
            seed=seed,
            min_train=int(cfg.get("min_train", 1)),
            min_val=int(cfg.get("min_val", 1)),
            min_test=int(cfg.get("min_test", 1)),
        )

    print(f"[INFO] Train={len(train_pairs)} Val={len(val_pairs)} Test={len(test_pairs)}", flush=True)

    num_workers = min(int(cfg.get("num_workers", 4)), os.cpu_count() or 4)
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_seed_worker,
    )

    train_loader = DataLoader(NiftiDeblurDataset(train_pairs),
                              batch_size=int(cfg.get("batch_size", 2)), shuffle=True,  **loader_kw)
    val_loader   = DataLoader(NiftiDeblurDataset(val_pairs),
                              batch_size=1, shuffle=False, **loader_kw)
    test_loader  = DataLoader(NiftiDeblurDataset(test_pairs),
                              batch_size=1, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader, test_pairs

    