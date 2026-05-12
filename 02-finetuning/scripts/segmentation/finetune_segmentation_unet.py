# finetune_segmentation_unet.py - Finetune pretrained LSM UNet backbone for segmentation

# --- Setup ---

# imports
import argparse
from datetime import datetime
import os
from pathlib import Path
import time

from monai.networks.nets import Unet

import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

# get shared finetuning utilities
from finetune_segmentation_utils import (
    FinetuneBaseModule,
    _seed_everything,
    _format_hms,
    build_dataloaders,
    load_config,
    run_inference,
)


# -------------------------
# UNet Lightning Module
# -------------------------

class FinetuneUNetModule(FinetuneBaseModule):

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

        self._freeze_encoder_epochs = int(freeze_encoder_epochs)
        self._encoder_frozen = self._freeze_encoder_epochs > 0

        # initialize shared loss, metrics, and logging state
        self._init_shared(loss_name=loss_name, freeze_bn_stats=freeze_bn_stats)

        # load pretrained weights
        if pretrained_ckpt is not None and str(pretrained_ckpt).strip() != "":
            self._load_pretrained(pretrained_ckpt)
        else:
            print("[INFO] Initializing UNet from scratch (no pretrained checkpoint).", flush=True)

    # freeze encoder for configured number of epochs then unfreeze
    # UNet decoder head is identified by the "model.2." prefix
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

    def configure_optimizers(self):
        base_lr = float(self.hparams.lr)
        wd = float(self.hparams.weight_decay)
        enc_mult = float(self.hparams.encoder_lr_mult)

        # UNet decoder head is "model.2.*"; everything else is encoder/backbone
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # seed
    seed = int(cfg.get("seed", 100))
    _seed_everything(seed)

    # paths
    out_root = Path(cfg["out_root"])
    pretrained_ckpt = cfg.get("pretrained_ckpt", None)

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

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Start: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)

    # build dataloaders
    train_loader, val_loader, train_pairs, val_pairs, test_pairs, val_tf, loader_kw = build_dataloaders(cfg, seed)

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
    if not best_ckpt or not os.path.exists(best_ckpt):
        trainer.save_checkpoint(str(ckpt_dir / "last.ckpt"))
        best_ckpt = str(ckpt_dir / "last.ckpt")
        print(f"[WARN] Best ckpt missing, using last: {best_ckpt}", flush=True)

    print(f"[INFO] Best checkpoint: {best_ckpt}", flush=True)

    # inference on test set (if provided)
    if test_pairs:
        infer_ckpt_choice = cfg.get("infer_ckpt", "best")
        infer_ckpt = best_ckpt if infer_ckpt_choice == "best" else str(ckpt_dir / "finetune_unet_best-v1.ckpt")
        run_inference(cfg, FinetuneUNetModule, infer_ckpt, test_pairs, val_tf, loader_kw, preds_dir, wandb_logger)

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)


if __name__ == "__main__":
    main()

    