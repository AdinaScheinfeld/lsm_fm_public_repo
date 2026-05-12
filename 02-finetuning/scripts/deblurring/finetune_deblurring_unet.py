# finetune_deblurring_unet.py - Finetune pretrained LSM UNet backbone for image deblurring

# --- Setup ---

# imports
import argparse
from datetime import datetime
import gc
import os
from pathlib import Path
import time

from monai.networks.nets import Unet

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch

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

# function to load UNet encoder weights from a pretraining checkpoint
# pretraining checkpoints store weights as student_encoder.<rest>
# we map to model.<rest> to match the UNet model structure
def _load_unet_backbone(unet: Unet, ckpt_path: str):
    print(f"[INFO] Loading pretrained UNet checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # map student_encoder.<rest> -> model.<rest>
    mapped = {}
    for k, v in sd.items():
        if k.startswith("student_encoder."):
            mapped["model." + k[len("student_encoder."):]] = v

    model_sd = unet.state_dict()
    safe, dropped = {}, []
    for k, v in mapped.items():
        if k not in model_sd:
            dropped.append((k, "not_in_model"))
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            dropped.append((k, f"shape mismatch {tuple(v.shape)} vs {tuple(model_sd[k].shape)}"))
            continue
        safe[k] = v

    incompat = unet.load_state_dict(safe, strict=False)
    print(
        f"[INFO] UNet load: kept={len(safe)} dropped={len(dropped)} "
        f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
        flush=True,
    )
    if dropped:
        print(f"[DEBUG] Dropped keys (first 10): {dropped[:10]}", flush=True)


# -------------------------
# UNet Lightning Module
# -------------------------

class FinetuneDeblurUNetModule(DeblurBaseModule):

    def __init__(
        self,
        pretrained_ckpt=None,
        lr: float = 1e-4,
        encoder_lr_mult: float = 0.1,
        freeze_encoder_epochs: int = 0,
        freeze_bn_stats: bool = False,
        weight_decay: float = 1e-5,
        unet_channels=(32, 64, 128, 256, 512),
        unet_strides=(2, 2, 2, 1),
        unet_num_res_units: int = 2,
        unet_norm: str = "BATCH",
        edge_weight: float = 0.1,
        highfreq_weight: float = 0.1,
        init: str = "pretrained",
    ):
        super().__init__()
        self.save_hyperparameters()

        # UNet backbone — predicts residual (same shape as input)
        self.model = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=tuple(int(x) for x in unet_channels),
            strides=tuple(int(x) for x in unet_strides),
            num_res_units=int(unet_num_res_units),
            norm=str(unet_norm),
        )

        # encoder freeze schedule
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_frozen = self.freeze_encoder_epochs > 0
        self._freeze_bn_stats = bool(freeze_bn_stats)

        # initialize shared loss state
        self._init_shared(edge_weight=edge_weight, highfreq_weight=highfreq_weight)

        # initialize weights
        if init == "pretrained":
            if not pretrained_ckpt:
                print("[WARN] init=pretrained but no pretrained_ckpt provided; using random init.", flush=True)
            else:
                try:
                    _load_unet_backbone(self.model, pretrained_ckpt)
                    if self.encoder_frozen:
                        for name, p in self.model.named_parameters():
                            if not name.startswith("model.2."):
                                p.requires_grad = False
                        print(f"[INFO] Encoder frozen for first {self.freeze_encoder_epochs} epochs.", flush=True)
                except Exception as e:
                    print(f"[ERROR] Failed to load pretrained checkpoint: {e} (using random init)", flush=True)
        else:
            print("[INFO] Using random initialization for UNet backbone.", flush=True)


    # *** Freezing Policy ***

    def on_train_start(self):
        # freeze BN running stats if configured (preserves pretrained batch statistics)
        if self._freeze_bn_stats:
            bn = 0
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    bn += 1
            print(f"[INFO] Froze {bn} BatchNorm layers.", flush=True)

    def on_train_epoch_start(self):
        # UNet decoder head is "model.2.*"; everything else is encoder/backbone
        if self.encoder_frozen and self.current_epoch < self.freeze_encoder_epochs:
            for name, p in self.model.named_parameters():
                if not name.startswith("model.2."):
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

        # UNet decoder head is "model.2.*"; everything else is encoder/backbone
        backbone, head = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (head if name.startswith("model.2.") else backbone).append(p)

        opt = torch.optim.AdamW(
            [{"params": backbone, "lr": base_lr * enc_mult},
             {"params": head,     "lr": base_lr}],
            weight_decay=wd,
            betas=(0.9, 0.999),
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

    # UNet architecture params
    unet_channels = tuple(int(x) for x in str(cfg.get("unet_channels", "32,64,128,256,512")).split(","))
    unet_strides  = tuple(int(x) for x in str(cfg.get("unet_strides",  "2,2,2,1")).split(","))

    # model
    model = FinetuneDeblurUNetModule(
        pretrained_ckpt=pretrained_ckpt,
        lr=float(cfg.get("lr", 1e-4)),
        encoder_lr_mult=float(cfg.get("encoder_lr_mult", 0.1)),
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)),
        freeze_bn_stats=bool(cfg.get("freeze_bn_stats", False)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        unet_channels=unet_channels,
        unet_strides=unet_strides,
        unet_num_res_units=int(cfg.get("unet_num_res_units", 2)),
        unet_norm=cfg.get("unet_norm", "BATCH"),
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
        filename="finetune_deblur_unet_best",
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
    infer_model = FinetuneDeblurUNetModule.load_from_checkpoint(best_ckpt, strict=False).to(device).eval()
    run_evaluation(infer_model, test_loader, test_pairs, preds_dir, device, wandb_logger=wandb_logger)

    del infer_model
    gc.collect()

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)


if __name__ == "__main__":
    main()

    