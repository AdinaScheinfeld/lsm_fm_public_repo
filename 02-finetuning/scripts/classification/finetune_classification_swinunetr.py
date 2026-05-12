# finetune_classification_swinunetr.py - Finetune pretrained LSM SwinUNETR backbone for patch classification

# --- Setup ---

# imports
import argparse
from datetime import datetime
import gc
import os
from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy

from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# get shared classification utilities
from finetune_classification_utils import (
    _seed_everything,
    _format_hms,
    build_dataloaders,
    load_config,
    run_evaluation,
    RunOutputs,
)


# -------------------------
# Checkpoint Loading
# -------------------------

# function to strip common wrapper prefixes from checkpoint state dict keys
def _strip_prefixes(state_dict, prefixes=("student_encoder.", "ema_student_encoder.", "module.", "net.", "encoder.")):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


# function to load swinViT encoder weights from a pretraining checkpoint into a SwinUNETR instance
def _load_swinunetr_backbone(model: SwinUNETR, ckpt_path: str):
    print(f"[INFO] Loading pretrained SwinUNETR checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # strip wrapper prefixes and keep only swinViT encoder keys
    sd = _strip_prefixes(sd)
    sd = {k: v for k, v in sd.items() if k.startswith("swinViT.")}

    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded = sum(1 for k in sd if k in model.state_dict())
    print(f"[INFO] SwinUNETR load: loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[DEBUG] Missing keys (first 10): {missing[:10]}", flush=True)
    return loaded, missing, unexpected


# -------------------------
# SwinUNETR Lightning Module
# -------------------------

class FinetuneSwinUNETRClassificationModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrained_ckpt=None,
        feature_size: int = 48,
        class_names=None,
        freeze_encoder_epochs: int = 0,
        linear_probe: bool = False,
        init_mode: str = "pretrained",
        in_channels: int = 1,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_names", "class_weights"])

        self.num_classes = int(num_classes)
        self.class_names = class_names or [str(i) for i in range(self.num_classes)]

        # SwinUNETR backbone — out_channels is not used for classification
        self.backbone = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=2,
            feature_size=feature_size,
            use_checkpoint=False,
        )

        # infer deepest encoder channel dim from a dummy forward pass through swinViT
        with torch.no_grad():
            dummy = torch.zeros((1, in_channels, 96, 96, 96))
            feats = self.backbone.swinViT(dummy)
            deepest = feats[-1] if isinstance(feats, (list, tuple)) else feats
            encoder_dim = deepest.shape[1]
        print(f"[INFO] SwinUNETR encoder deepest C={encoder_dim} (feature_size={feature_size})", flush=True)

        # global average pooling + classification head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(encoder_dim, self.num_classes)

        # loss function with optional class weighting
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # freeze policy
        self._freeze_encoder_epochs = int(freeze_encoder_epochs)
        self._linear_probe = bool(linear_probe)

        # initialize weights
        if init_mode not in ("pretrained", "random"):
            raise ValueError("init_mode must be 'pretrained' or 'random'")

        if init_mode == "pretrained":
            if not pretrained_ckpt:
                print("[WARN] init_mode=pretrained but no pretrained_ckpt provided; using random init.", flush=True)
            else:
                try:
                    _load_swinunetr_backbone(self.backbone, pretrained_ckpt)
                except Exception as e:
                    print(f"[ERROR] Failed to load pretrained checkpoint: {e} (using random init)", flush=True)
        else:
            print("[INFO] Using random initialization for SwinUNETR backbone.", flush=True)

        # metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy   = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_accuracy  = MulticlassAccuracy(num_classes=self.num_classes)


    # *** Freezing Policy ***

    def _set_encoder_requires_grad(self, flag: bool):
        for p in self.backbone.swinViT.parameters():
            p.requires_grad = flag

    def on_train_start(self):
        if self._linear_probe:
            print("[INFO] Linear probe: freezing encoder for entire training.", flush=True)
            self._set_encoder_requires_grad(False)
        elif self._freeze_encoder_epochs > 0 and self.current_epoch < self._freeze_encoder_epochs:
            print(f"[INFO] Freezing encoder for epoch {self.current_epoch}/{self._freeze_encoder_epochs}.", flush=True)
            self._set_encoder_requires_grad(False)

    def on_train_epoch_start(self):
        if self._linear_probe:
            return
        if self._freeze_encoder_epochs > 0:
            still_freeze = self.current_epoch < self._freeze_encoder_epochs
            self._set_encoder_requires_grad(not still_freeze)
            if still_freeze:
                print(f"[INFO] Freezing encoder for epoch {self.current_epoch}/{self._freeze_encoder_epochs}.", flush=True)


    # *** Forward Pass ***

    def _encode(self, x):
        # run swinViT and take the deepest feature map
        feats = self.backbone.swinViT(x)
        return feats[-1] if isinstance(feats, (list, tuple)) else feats

    def forward(self, x):
        z = self._encode(x)         # [B, C, D, H, W]
        z = self.pool(z)            # [B, C, 1, 1, 1]
        z = torch.flatten(z, 1)    # [B, C]
        return self.head(z)         # [B, num_classes]


    # *** Steps ***

    def _step(self, batch, stage: str):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.train_accuracy.update(preds, y)
            self.log("train_loss",     loss,                 prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_accuracy", self.train_accuracy,  prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "val":
            self.val_accuracy.update(preds, y)
            self.log("val_loss",     loss,               prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            self.log("val_accuracy", self.val_accuracy,  prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        elif stage == "test":
            self.test_accuracy.update(preds, y)
            self.log("test_loss",     loss,                prog_bar=True, on_step=False, on_epoch=True)
            self.log("test_accuracy", self.test_accuracy,  prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")


    # *** Optimizer ***

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))


# -------------------------
# Main
# -------------------------

def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()
    cfg = load_config(args.config)

    # seed
    seed = int(cfg.get("seed", 100))
    _seed_everything(seed)

    # output dirs
    out_root = Path(cfg["out_root"])
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
    train_loader, val_loader, test_loader, class_names, class_weights = build_dataloaders(cfg, seed)
    K = len(class_names)

    # wandb
    wandb_project = cfg.get("wandb_project", None)
    run_name = cfg.get("run_name", None)
    wandb_logger = WandbLogger(project=wandb_project, name=run_name) if wandb_project else None

    # pretrained checkpoint
    pretrained_ckpt = cfg.get("pretrained_ckpt", None)
    if cfg.get("init", "pretrained") == "scratch":
        pretrained_ckpt = None
        print("[INFO] init=scratch: ignoring pretrained_ckpt.", flush=True)

    # model
    model = FinetuneSwinUNETRClassificationModule(
        num_classes=K,
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        pretrained_ckpt=pretrained_ckpt,
        feature_size=int(cfg.get("swinunetr_feature_size", 48)),
        class_names=class_names,
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)),
        linear_probe=bool(cfg.get("linear_probe", False)),
        init_mode=cfg.get("init", "pretrained"),
        in_channels=int(cfg.get("in_channels", 1)),
        class_weights=class_weights,
    )

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="finetune_cls_swinunetr_best",
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=int(cfg.get("early_stopping_patience", 50)),
    )

    trainer = pl.Trainer(
        max_epochs=int(cfg.get("max_epochs", 200)),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        log_every_n_steps=10,
        deterministic=True,
    )

    # train
    print(f"[INFO] Starting training: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val patches, {K} classes.", flush=True)
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = model_ckpt.best_model_path
    if not best_ckpt or not os.path.exists(best_ckpt):
        trainer.save_checkpoint(str(ckpt_dir / "last.ckpt"))
        best_ckpt = str(ckpt_dir / "last.ckpt")
        print(f"[WARN] Best ckpt missing, using last: {best_ckpt}", flush=True)

    print(f"[INFO] Best checkpoint: {best_ckpt}", flush=True)

    # inference on test set
    infer_model = FinetuneSwinUNETRClassificationModule.load_from_checkpoint(best_ckpt, strict=False).to(device).eval()
    preds_csv = run_evaluation(infer_model, test_loader, class_names, device, preds_dir, tag="swinunetr", wandb_logger=wandb_logger)

    del infer_model
    gc.collect()

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)


if __name__ == "__main__":
    main()

    