# finetune_deblurring_swinunetr.py - Finetune pretrained LSM SwinUNETR backbone for image deblurring

# --- Setup ---

# imports
import argparse
from datetime import datetime
import gc
import os
from pathlib import Path
import time

from monai.data.meta_tensor import MetaTensor
from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
from torch.serialization import add_safe_globals

# get shared deblurring utilities
from finetune_deblurring_utils import (
    DeblurBaseModule,
    _seed_everything,
    _format_hms,
    build_dataloaders,
    load_config,
    run_evaluation,
)


# -------------------------
# Checkpoint Loading
# -------------------------

# function to load swinViT encoder weights from a pretraining checkpoint into a SwinUNETR instance
def _load_swinunetr_backbone(model: SwinUNETR, ckpt_path: str):
    print(f"[INFO] Loading pretrained SwinUNETR checkpoint: {ckpt_path}", flush=True)
    add_safe_globals([MetaTensor])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # strip known wrapper prefixes and keep only swinViT encoder keys
    new_sd = {}
    for k, v in sd.items():
        for pfx in ("model.", "module.", "student_model.", "student.", "teacher_model.", "net.", "encoder.", "student_encoder.", "backbone."):
            if k.startswith(pfx):
                k = k[len(pfx):]
        if k.startswith("swinViT."):
            new_sd[k] = v
        elif "swinViT." in k:
            _, sub = k.split("swinViT.", 1)
            new_sd["swinViT." + sub] = v

    # drop any keys with incompatible shapes
    for key in ("swinViT.patch_embed.proj.weight", "swinViT.patch_embed.proj.bias"):
        if key in new_sd and new_sd[key].shape != model.state_dict().get(key, new_sd[key]).shape:
            print(f"[WARN] Dropping incompatible key {key}: ckpt {new_sd[key].shape} vs model {model.state_dict()[key].shape}", flush=True)
            new_sd.pop(key)

    incompat = model.load_state_dict(new_sd, strict=False)
    print(f"[INFO] SwinUNETR load: kept={len(new_sd)} missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}", flush=True)


# -------------------------
# SwinUNETR Lightning Module
# -------------------------

class FinetuneDeblurSwinUNETRModule(DeblurBaseModule):

    def __init__(
        self,
        pretrained_ckpt=None,
        lr: float = 1e-4,
        feature_size: int = 48,
        encoder_lr_mult: float = 0.05,
        freeze_encoder_epochs: int = 0,
        weight_decay: float = 1e-5,
        edge_weight: float = 0.1,
        highfreq_weight: float = 0.1,
        init: str = "pretrained",
    ):
        super().__init__()
        self.save_hyperparameters()

        # SwinUNETR backbone — predicts residual (same shape as input)
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True,
        )

        # encoder freeze schedule
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_frozen = self.freeze_encoder_epochs > 0

        # initialize shared loss state
        self._init_shared(edge_weight=edge_weight, highfreq_weight=highfreq_weight)

        # initialize weights
        if init == "pretrained":
            if not pretrained_ckpt:
                print("[WARN] init=pretrained but no pretrained_ckpt provided; using random init.", flush=True)
            else:
                try:
                    _load_swinunetr_backbone(self.model, pretrained_ckpt)
                    if self.encoder_frozen:
                        for name, p in self.model.named_parameters():
                            if name.startswith("swinViT."):
                                p.requires_grad = False
                        print(f"[INFO] Encoder frozen for first {self.freeze_encoder_epochs} epochs.", flush=True)
                except Exception as e:
                    print(f"[ERROR] Failed to load pretrained checkpoint: {e} (using random init)", flush=True)
        else:
            print("[INFO] Using random initialization for SwinUNETR backbone.", flush=True)


    # *** Freezing Policy ***

    def on_train_epoch_start(self):
        if self.encoder_frozen and self.current_epoch < self.freeze_encoder_epochs:
            for name, p in self.model.named_parameters():
                if name.startswith("swinViT."):
                    p.requires_grad = False
        elif self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            print(f"[INFO] Unfreezing encoder at epoch {self.current_epoch}.", flush=True)
            for _, p in self.model.named_parameters():
                p.requires_grad = True
            self.encoder_frozen = False


    # *** Optimizer ***

    def configure_optimizers(self):
        base_lr  = float(self.hparams.lr)
        wd       = float(self.hparams.weight_decay)
        enc_mult = float(self.hparams.encoder_lr_mult)

        # separate encoder (swinViT) and decoder parameters for differential learning rates
        encoder_params, decoder_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (encoder_params if name.startswith("swinViT.") else decoder_params).append(p)

        opt = torch.optim.AdamW(
            [{"params": encoder_params, "lr": base_lr * enc_mult},
             {"params": decoder_params, "lr": base_lr}],
            weight_decay=wd,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=int(self.trainer.max_epochs), eta_min=base_lr * 0.1
        )
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # seed
    seed = int(cfg.get("seed", 100))
    _seed_everything(seed)

    # output dirs
    out_root  = Path(cfg["out_root"])
    ckpt_dir  = out_root / "checkpoints"
    preds_dir = out_root / "preds"
    logs_dir  = out_root / "logs"
    for d in [ckpt_dir, preds_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.perf_counter()
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Start: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

    # build dataloaders
    train_loader, val_loader, test_loader, test_pairs = build_dataloaders(cfg, seed)

    # wandb
    wandb_project = cfg.get("wandb_project", None)
    run_name      = cfg.get("run_name", None)
    wandb_logger  = WandbLogger(project=wandb_project, name=run_name) if wandb_project else None

    # pretrained checkpoint
    pretrained_ckpt = cfg.get("pretrained_ckpt", None)
    if cfg.get("init", "pretrained") == "scratch":
        pretrained_ckpt = None
        print("[INFO] init=scratch: ignoring pretrained_ckpt.", flush=True)

    # model
    model = FinetuneDeblurSwinUNETRModule(
        pretrained_ckpt=pretrained_ckpt,
        lr=float(cfg.get("lr", 1e-4)),
        feature_size=int(cfg.get("swinunetr_feature_size", 48)),
        encoder_lr_mult=float(cfg.get("encoder_lr_mult", 0.05)),
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        edge_weight=float(cfg.get("edge_weight", 0.1)),
        highfreq_weight=float(cfg.get("highfreq_weight", 0.1)),
        init=cfg.get("init", "pretrained"),
    )

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="finetune_deblur_swinunetr_best",
    )
    early_stopping = EarlyStopping(
        monitor="val_total_loss",
        mode="min",
        patience=int(cfg.get("early_stopping_patience", 50)),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.get("max_epochs", 500)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        log_every_n_steps=1,
        deterministic=True,
    )

    # train
    print(f"[INFO] Starting training: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val pairs.", flush=True)
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = model_ckpt.best_model_path
    if not best_ckpt or not os.path.exists(best_ckpt):
        trainer.save_checkpoint(str(ckpt_dir / "last.ckpt"))
        best_ckpt = str(ckpt_dir / "last.ckpt")
        print(f"[WARN] Best ckpt missing, using last: {best_ckpt}", flush=True)

    print(f"[INFO] Best checkpoint: {best_ckpt}", flush=True)

    # inference on test set
    infer_model = FinetuneDeblurSwinUNETRModule.load_from_checkpoint(best_ckpt, strict=False).to(device).eval()
    run_evaluation(infer_model, test_loader, test_pairs, preds_dir, device, wandb_logger=wandb_logger)

    del infer_model
    gc.collect()

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)


if __name__ == "__main__":
    main()

    