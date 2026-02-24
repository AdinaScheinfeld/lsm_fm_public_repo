# finetune_segmentation_unet.py - Finetune pretrained LSM UNet backbone for segmentation

# --- Setup ---

# imports
import argparse
import csv
import gc
from dataclasses import dataclass
from datetime import datetime
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import random
import sys
import time
import yaml
import wandb

from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import Unet

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# get augmentation functions
sys.path.append('../../00-helpers/')
from all_datasets_transforms import get_finetune_train_transforms, get_finetune_val_transforms

torch.set_float32_matmul_precision("medium")


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
        random.seed(base_seed + info.id)
        np.random.seed(base_seed + info.id)

# helper to format seconds as H:MM:SS
def _format_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}s"


# dataclass to hold image-label pairs
@dataclass
class Pair:
    image: Path
    label: Path


# function to discover image-label pairs in a directory
# looks for .nii/.nii.gz files that have a corresponding _label file
# optionally filters by channel substring (e.g. "ch0", "ch1", or "ALL" for no filtering)
def discover_pairs(data_dir: Path, channel_substr="ALL", file_prefix=None):

    substrings = None
    if channel_substr:
        s = str(channel_substr).strip()
        if s and s.upper() != "ALL":
            substrings = [t.strip().lower() for t in s.split(",") if t.strip()]

    # normalize prefix filter
    prefix = str(file_prefix).strip().lower() if file_prefix else None
    print(f"[INFO] discover_pairs: data_dir={data_dir} channel_substr={channel_substr} file_prefix={file_prefix} -> substrings={substrings} prefix={prefix}", flush=True)

    pairs = []
    for p in sorted(data_dir.glob("*.nii*")):
        lower = p.name.lower()
        if lower.endswith("_label.nii") or lower.endswith("_label.nii.gz"):
            continue
        if prefix and not lower.startswith(prefix):
            continue
        if substrings is not None and not any(sub in lower for sub in substrings):
            continue
        suffix = "".join(p.suffixes)
        base = p.name[:-len(suffix)]
        label = p.with_name(f"{base}_label{suffix}")
        if label.exists():
            pairs.append(Pair(image=p, label=label))

    return pairs


# function to split pairs into train and val sets
def split_train_val(pairs, val_percent=0.2, seed=100, min_train=1, min_val=1):
    pairs = list(pairs)
    n = len(pairs)
    if n < (min_train + min_val):
        raise ValueError(f"Not enough pairs ({n}) for min_train={min_train} + min_val={min_val}.")
    rng = random.Random(seed + 1)
    rng.shuffle(pairs)
    n_val = max(min_val, min(int(round(n * val_percent)), n - min_train))
    n_train = n - n_val
    if n_train < min_train or n_val < min_val:
        raise ValueError(f"Split failed: train={n_train} val={n_val} with min_train={min_train} min_val={min_val}.")
    return pairs[n_val:], pairs[:n_val]


# function to run model inference on a single input tensor
@torch.no_grad()
def predict_logits(model, x):
    model.eval()
    device = next(model.parameters()).device
    return model(x.to(device))


# function to compute Dice at a given threshold from model logits
def dice_at_threshold_from_logits(logits, target, threshold=0.5, eps=1e-8):
    if target.device != logits.device:
        target = target.to(logits.device)
    target = target.to(dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(dtype=logits.dtype)
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum() + eps
    return float((2.0 * inter / denom).item())


# function to save binary prediction mask as NIfTI, copying affine/header from the original image
def save_pred_nii(mask_bin, like_path, out_path):
    vol = mask_bin.squeeze().detach().cpu().numpy().astype(np.uint8)
    try:
        like = nib.load(str(like_path))
        affine, header = like.affine, like.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol, affine, header), str(out_path))


# function to save probability map as NIfTI
def save_prob_nii(probs, like_path, out_path):
    vol = probs.squeeze().detach().cpu().numpy().astype(np.float32)
    try:
        like = nib.load(str(like_path))
        affine, header = like.affine, like.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol, affine, header), str(out_path))


# -------------------------
# Dataset
# -------------------------

# dataset to load NIfTI image-label pairs and apply MONAI transforms
class NiftiPairDictDataset(Dataset):

    def __init__(self, pairs, transform=None):
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = nib.load(str(p.image)).get_fdata().astype(np.float32)
        lbl = nib.load(str(p.label)).get_fdata().astype(np.float32)

        # remove trailing singleton dim if present
        if img.ndim == 4 and img.shape[-1] == 1:
            img = img[..., 0]
        if lbl.ndim == 4 and lbl.shape[-1] == 1:
            lbl = lbl[..., 0]

        # add channel dim and binarize label
        img = img[None, ...]
        lbl = (lbl[None, ...] > 0.5).astype(np.float32)

        d = {"image": img, "label": lbl, "filename": str(p.image)}
        if self.transform is not None:
            d = self.transform(d)
        return d


# -------------------------
# Lightning Module
# -------------------------

class FinetuneUNetModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_ckpt,
        lr,
        weight_decay,
        loss_name,
        encoder_lr_mult,
        freeze_encoder_epochs,
        freeze_bn_stats,
        channels,
        strides,
        num_res_units,
        norm,
    ):
        super().__init__()
        self.save_hyperparameters()

        # UNet with out_channels=1 for binary segmentation
        self.model = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=int(num_res_units),
            norm=str(norm),
        )

        # loss function
        ln = str(loss_name).lower()
        if ln == "dicefocal":
            self.loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=0.5, lambda_focal=0.5, alpha=0.25, gamma=2.0)
        else:
            self.loss_fn = DiceCELoss(sigmoid=True)

        self._freeze_encoder_epochs = int(freeze_encoder_epochs)
        self._encoder_frozen = self._freeze_encoder_epochs > 0
        self._freeze_bn_stats = bool(int(freeze_bn_stats))

        self.val_dice = DiceMetric(include_background=False, reduction="mean")

        # wandb image logging state
        self._val_table = None
        self._val_logged = 0
        self._val_max_log = 5

        # load pretrained weights
        if pretrained_ckpt is not None and str(pretrained_ckpt).strip() != "":
            self._load_pretrained(pretrained_ckpt)
        else:
            print("[INFO] Initializing UNet from scratch (no pretrained checkpoint).", flush=True)

    # load pretrained checkpoint and map student_encoder.* -> model.*
    def _load_pretrained(self, ckpt_path):
        print(f"[INFO] Loading pretrained checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        mapped = {}
        for k, v in sd.items():
            if k.startswith("student_encoder."):
                mapped["model." + k[len("student_encoder."):]] = v

        model_sd = self.state_dict()
        safe, dropped = {}, []
        for k, v in mapped.items():
            if k not in model_sd:
                dropped.append((k, "not_in_model"))
                continue
            if tuple(v.shape) != tuple(model_sd[k].shape):
                dropped.append((k, f"shape mismatch {tuple(v.shape)} vs {tuple(model_sd[k].shape)}"))
                continue
            safe[k] = v

        incompat = self.load_state_dict(safe, strict=False)
        print(
            f"[INFO] Pretrained load: kept={len(safe)} dropped={len(dropped)} "
            f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
            flush=True,
        )

    def forward(self, x):
        return self.model(x)

    # freeze BN stats if configured
    def on_train_start(self):
        if self._freeze_bn_stats:
            bn = 0
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    bn += 1
            print(f"[INFO] Froze {bn} BatchNorm layers.", flush=True)

    # create fresh wandb table at the start of each val epoch
    def on_validation_epoch_start(self):
        self._val_logged = 0
        if isinstance(getattr(self, "logger", None), WandbLogger):
            self._val_table = wandb.Table(columns=["filename", "image_mid", "label_mid", "pred_mid"])
        else:
            self._val_table = None

    # freeze encoder for configured number of warmup epochs then unfreeze
    def on_train_epoch_start(self):
        if self._encoder_frozen and self.current_epoch < self._freeze_encoder_epochs:
            for name, p in self.model.named_parameters():
                if not name.startswith("model.2."):
                    p.requires_grad = False
        elif self._encoder_frozen and self.current_epoch >= self._freeze_encoder_epochs:
            print(f"[INFO] Unfreezing encoder at epoch {self.current_epoch}.", flush=True)
            for _, p in self.model.named_parameters():
                p.requires_grad = True
            self._encoder_frozen = False

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float() # Dice@0.5
        self.val_dice(pred, y)

        # log sample images to wandb table
        if self._val_table is not None and self._val_logged < self._val_max_log:
            img = x[0, 0].detach().float().cpu().numpy()
            lbl = y[0, 0].detach().float().cpu().numpy()
            pred_bin = (torch.sigmoid(logits[0, 0]).detach().float().cpu().numpy() > 0.5).astype(np.float32)
            mid = img.shape[0] // 2
            fname = ""
            if isinstance(batch.get("filename", None), (list, tuple)):
                fname = str(batch["filename"][0])
            self._val_table.add_data(fname, wandb.Image(img[mid]), wandb.Image(lbl[mid]), wandb.Image(pred_bin[mid]))
            self._val_logged += 1

        return loss

    def on_validation_epoch_end(self):
        d = self.val_dice.aggregate().item()
        self.val_dice.reset()
        self.log("val_dice_050", float(d), on_step=False, on_epoch=True, prog_bar=True)

        if self._val_table is not None and self._val_logged > 0 and isinstance(getattr(self, "logger", None), WandbLogger):
            self.logger.experiment.log({f"val_samples_epoch_{self.current_epoch}": self._val_table}, commit=False)
        self._val_table = None

    def configure_optimizers(self):
        base_lr = float(self.hparams.lr)
        wd = float(self.hparams.weight_decay)
        enc_mult = float(self.hparams.encoder_lr_mult)

        backbone, head = [], []
        for name, p in self.model.named_parameters():
            if name.startswith("model.2."):
                head.append(p)
            else:
                backbone.append(p)

        return torch.optim.AdamW(
            [{"params": backbone, "lr": base_lr * enc_mult},
             {"params": head, "lr": base_lr}],
            weight_decay=wd,
            betas=(0.9, 0.999),
        )


# -------------------------
# Main
# -------------------------

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # seed
    seed = int(cfg.get("seed", 100))
    _seed_everything(seed)

    # paths
    train_dir = Path(cfg["train_dir"])
    val_dir = cfg.get("val_dir", None)
    test_dir = cfg.get("test_dir", None)
    out_root = Path(cfg["out_root"])
    pretrained_ckpt = cfg.get("pretrained_ckpt", None)
    channel_substr = cfg.get("channel_substr", "ALL")

    # if init is scratch, ignore any pretrained ckpt
    if cfg.get("init", "pretrained") == "scratch":
        pretrained_ckpt = None
        print("[INFO] init=scratch: ignoring pretrained_ckpt.", flush=True)

    # create output dirs
    ckpt_dir = out_root / "checkpoints"
    preds_dir = out_root / "preds"
    logs_dir = out_root / "logs"
    for d in [ckpt_dir, preds_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Start: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

    # discover pairs
    file_prefix = cfg.get("file_prefix", None)
    train_pairs_raw = discover_pairs(train_dir, channel_substr=channel_substr, file_prefix=file_prefix)
    print(f"[INFO] Found {len(train_pairs_raw)} pairs in train_dir: {train_dir}", flush=True)

    # if a separate val_dir is provided, use it directly; otherwise split train_dir
    if val_dir is not None:
        val_pairs = discover_pairs(Path(val_dir), channel_substr=channel_substr, file_prefix=file_prefix)
        train_pairs = train_pairs_raw
        print(f"[INFO] Using separate val_dir: {val_dir} ({len(val_pairs)} pairs)", flush=True)
    else:
        val_percent = float(cfg.get("val_percent", 0.2))
        train_pairs, val_pairs = split_train_val(
            train_pairs_raw,
            val_percent=val_percent,
            seed=seed,
            min_train=int(cfg.get("min_train", 1)),
            min_val=int(cfg.get("min_val", 1)),
        )
        print(f"[INFO] Split {len(train_pairs_raw)} pairs -> train={len(train_pairs)} val={len(val_pairs)} (val_percent={val_percent})", flush=True)

    # test pairs (optional)
    if test_dir is not None:
        test_pairs = discover_pairs(Path(test_dir), channel_substr=channel_substr, file_prefix=file_prefix)
        print(f"[INFO] Using test_dir: {test_dir} ({len(test_pairs)} pairs)", flush=True)
    else:
        test_pairs = []
        print("[INFO] No test_dir provided — skipping inference.", flush=True)

    # transforms and dataloaders
    train_tf = get_finetune_train_transforms()
    val_tf = get_finetune_val_transforms()

    num_workers = min(int(cfg.get("num_workers", 4)), os.cpu_count() or 4)
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_seed_worker,
    )

    train_loader = DataLoader(NiftiPairDictDataset(train_pairs, transform=train_tf),
                              batch_size=int(cfg.get("batch_size", 2)), shuffle=True, **loader_kw)
    val_loader = DataLoader(NiftiPairDictDataset(val_pairs, transform=val_tf),
                            batch_size=1, shuffle=False, **loader_kw)

    # wandb
    wandb_project = cfg.get("wandb_project", None)
    run_name = cfg.get("run_name", None)
    wandb_logger = WandbLogger(project=wandb_project, name=run_name) if wandb_project else None

    # model
    unet_channels = tuple(int(x) for x in str(cfg.get("unet_channels", "32,64,128,256,512")).split(","))
    unet_strides = tuple(int(x) for x in str(cfg.get("unet_strides", "2,2,2,1")).split(","))

    model = FinetuneUNetModule(
        pretrained_ckpt=pretrained_ckpt,
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        loss_name=cfg.get("loss_name", "dicefocal"),
        encoder_lr_mult=float(cfg.get("encoder_lr_mult", 0.2)),
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 10)),
        freeze_bn_stats=int(cfg.get("freeze_bn_stats", 1)),
        channels=unet_channels,
        strides=unet_strides,
        num_res_units=int(cfg.get("unet_num_res_units", 2)),
        norm=cfg.get("unet_norm", "BATCH"),
    )

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="finetune_unet_best",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(cfg.get("early_stopping_patience", 200)),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.get("max_epochs", 600)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        log_every_n_steps=10,
        deterministic=True,
    )

    # train
    print(f"[INFO] Starting training: {len(train_pairs)} train, {len(val_pairs)} val pairs.", flush=True)
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = model_ckpt.best_model_path
    last_ckpt = str(ckpt_dir / "finetune_unet_best-v1.ckpt")
    if not best_ckpt or not os.path.exists(best_ckpt):
        trainer.save_checkpoint(str(ckpt_dir / "last.ckpt"))
        best_ckpt = str(ckpt_dir / "last.ckpt")
        print(f"[WARN] Best ckpt missing, using last: {best_ckpt}", flush=True)

    print(f"[INFO] Best checkpoint: {best_ckpt}", flush=True)

    # inference on test set (if provided)
    if test_pairs:
        infer_ckpt_choice = cfg.get("infer_ckpt", "best")
        infer_ckpt = best_ckpt if infer_ckpt_choice == "best" else last_ckpt
        print(f"[INFO] Running inference with ckpt: {infer_ckpt}", flush=True)

        infer_model = FinetuneUNetModule.load_from_checkpoint(infer_ckpt).to(device).eval()
        test_loader = DataLoader(NiftiPairDictDataset(test_pairs, transform=val_tf),
                                 batch_size=1, shuffle=False, **loader_kw)

        rows = []
        for batch in test_loader:
            x = batch["image"]
            y = batch["label"]
            fname = Path(batch["filename"][0])

            logits = predict_logits(infer_model, x)
            dice_050 = dice_at_threshold_from_logits(logits, y, threshold=0.5)

            probs = torch.sigmoid(logits)
            mask_bin = (probs >= 0.5).to(torch.uint8)

            base_stem = fname.stem.replace(".nii", "").replace(".gz", "")
            pred_path = preds_dir / f"{base_stem}_pred.nii.gz"
            prob_path = preds_dir / f"{base_stem}_prob.nii.gz"
            save_pred_nii(mask_bin, like_path=fname, out_path=pred_path)
            save_prob_nii(probs, like_path=fname, out_path=prob_path)

            rows.append({
                "filename": fname.name,
                "image_path": str(fname),
                "dice_050": f"{dice_050:.6f}",
                "pred_path": str(pred_path),
                "prob_path": str(prob_path),
            })
            print(f"[INFO] {fname.name}  Dice@0.5={dice_050:.6f}", flush=True)

        if rows:
            mean_dice = float(np.mean([float(r["dice_050"]) for r in rows]))
            rows.append({"filename": "MEAN", "image_path": "", "dice_050": f"{mean_dice:.6f}", "pred_path": "", "prob_path": ""})
            print(f"[INFO] Mean test Dice@0.5 = {mean_dice:.6f}", flush=True)
            if wandb_logger:
                wandb_logger.experiment.summary["test_mean_dice_050"] = mean_dice

        metrics_csv = preds_dir / "metrics_test.csv"
        with open(metrics_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "image_path", "dice_050", "pred_path", "prob_path"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Metrics saved to: {metrics_csv}", flush=True)

        del infer_model, test_loader
        gc.collect()

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)


if __name__ == "__main__":
    main()



