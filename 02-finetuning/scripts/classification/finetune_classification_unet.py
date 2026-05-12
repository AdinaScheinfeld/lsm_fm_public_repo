# finetune_classification_unet.py - Finetune pretrained LSM UNet backbone for patch classification

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

from monai.networks.nets import Unet

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
def _strip_prefixes(state_dict, prefixes=("student_encoder.", "ema_student_encoder.", "model.", "module.", "net.", "encoder.")):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


# function to load UNet encoder weights from a pretraining checkpoint
# pretraining checkpoints store weights as student_encoder.<rest>
# we try mapping to model.<rest> first (matching the segmentation loader convention), then fall back to <rest> directly
def _load_unet_backbone(unet: Unet, ckpt_path: str):
    print(f"[INFO] Loading pretrained UNet checkpoint: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # keep only student_encoder keys, then strip prefixes
    student = {k: v for k, v in sd.items() if k.startswith("student_encoder.")}
    if not student:
        student = sd  # fallback: checkpoint saved without student_encoder prefix
    student = _strip_prefixes(student, prefixes=("student_encoder.", "ema_student_encoder.", "module.", "net.", "encoder."))

    model_sd = unet.state_dict()

    # attempt A: map to "model.<rest>" (matches segmentation finetuning convention)
    mapped_a = {f"model.{k}": v for k, v in student.items()}
    safe_a = {k: v for k, v in mapped_a.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}

    # attempt B: map to "<rest>" directly
    safe_b = {k: v for k, v in student.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}

    # use whichever mapping loads more weights
    safe = safe_a if len(safe_a) >= len(safe_b) else safe_b
    strategy = "model.<rest>" if len(safe_a) >= len(safe_b) else "<rest>"

    missing, unexpected = unet.load_state_dict(safe, strict=False)
    loaded = sum(1 for k in safe if k in model_sd)
    print(f"[INFO] UNet load ({strategy}): loaded={loaded} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print(f"[DEBUG] Missing keys (first 10): {missing[:10]}", flush=True)
    return loaded, missing, unexpected


# -------------------------
# UNet Lightning Module
# -------------------------

class FinetuneUNetClassificationModule(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrained_ckpt=None,
        class_names=None,
        freeze_encoder_epochs: int = 0,
        linear_probe: bool = False,
        init_mode: str = "pretrained",
        in_channels: int = 1,
        unet_channels=(32, 64, 128, 256, 512),
        unet_strides=(2, 2, 2, 1),
        unet_num_res_units: int = 2,
        unet_norm: str = "BATCH",
        class_weights=None,
        img_size=(96, 96, 96),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_names", "class_weights"])

        self.num_classes = int(num_classes)
        self.class_names = class_names or [str(i) for i in range(self.num_classes)]

        # UNet backbone
        self.backbone = Unet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,  # out_channels not used for classification; bottleneck features are extracted via hook
            channels=tuple(int(x) for x in unet_channels),
            strides=tuple(int(x) for x in unet_strides),
            num_res_units=int(unet_num_res_units),
            norm=str(unet_norm),
        )

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
                    _load_unet_backbone(self.backbone, pretrained_ckpt)
                except Exception as e:
                    print(f"[ERROR] Failed to load pretrained checkpoint: {e} (using random init)", flush=True)
        else:
            print("[INFO] Using random initialization for UNet backbone.", flush=True)

        # auto-detect bottleneck layer via forward hooks
        # selects the 5D activation with the smallest spatial volume (deepest feature map),
        # breaking ties by largest number of channels
        self._bottleneck_name = None
        self._bottleneck_channels = None
        self._capture = None
        self._infer_bottleneck(in_channels=in_channels, img_size=img_size)

        # pooling + classification head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(int(self._bottleneck_channels), self.num_classes)

        # metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy   = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_accuracy  = MulticlassAccuracy(num_classes=self.num_classes)


    # *** Bottleneck Detection ***

    def _infer_bottleneck(self, in_channels: int, img_size):
        candidates = []

        def make_hook(name):
            def hook(_m, _inp, out):
                if torch.is_tensor(out) and out.ndim == 5:
                    B, C, D, H, W = out.shape
                    candidates.append((int(D) * int(H) * int(W), int(C), name))
            return hook

        hooks = []
        for name, m in self.backbone.named_modules():
            hooks.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            _ = self.backbone(torch.zeros((1, in_channels, *img_size)))

        for h in hooks:
            h.remove()

        if not candidates:
            # fallback: use UNet output directly
            print("[WARN] No 5D activations captured; falling back to UNet output as features.", flush=True)
            self._bottleneck_name = "__UNET_OUTPUT__"
            with torch.no_grad():
                y = self.backbone(torch.zeros((1, in_channels, *img_size)))
            self._bottleneck_channels = int(y.shape[1])
            print(f"[INFO] Bottleneck fallback channels={self._bottleneck_channels}", flush=True)
            return

        # choose the activation with smallest spatial volume, largest channels as tiebreaker
        candidates.sort(key=lambda t: (t[0], -t[1]))
        spatial, C, name = candidates[0]
        self._bottleneck_name = name
        self._bottleneck_channels = C
        print(f"[INFO] UNet bottleneck: module='{name}' spatial={spatial} C={C}", flush=True)

        # attach persistent hook for use during forward
        self._capture = {"z": None}

        def persistent_hook(_m, _inp, out):
            if torch.is_tensor(out) and out.ndim == 5:
                self._capture["z"] = out

        for n, m in self.backbone.named_modules():
            if n == name:
                m.register_forward_hook(persistent_hook)
                break


    # *** Freezing Policy ***

    def _set_backbone_requires_grad(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad = flag

    def on_train_start(self):
        if self._linear_probe:
            print("[INFO] Linear probe: freezing backbone for entire training.", flush=True)
            self._set_backbone_requires_grad(False)
        elif self._freeze_encoder_epochs > 0 and self.current_epoch < self._freeze_encoder_epochs:
            print(f"[INFO] Freezing backbone for epoch {self.current_epoch}/{self._freeze_encoder_epochs}.", flush=True)
            self._set_backbone_requires_grad(False)

    def on_train_epoch_start(self):
        if self._linear_probe:
            return
        if self._freeze_encoder_epochs > 0:
            still_freeze = self.current_epoch < self._freeze_encoder_epochs
            self._set_backbone_requires_grad(not still_freeze)
            if still_freeze:
                print(f"[INFO] Freezing backbone for epoch {self.current_epoch}/{self._freeze_encoder_epochs}.", flush=True)


    # *** Forward Pass ***

    def _encode(self, x):
        if self._capture is not None:
            self._capture["z"] = None
        _ = self.backbone(x)

        if self._bottleneck_name == "__UNET_OUTPUT__":
            return _

        z = self._capture.get("z", None) if self._capture is not None else None
        if z is None:
            return _  # fallback if hook didn't fire
        return z

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
            self.log("train_loss",     loss,                prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_accuracy", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
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

    # UNet architecture params
    unet_channels   = tuple(int(x) for x in str(cfg.get("unet_channels", "32,64,128,256,512")).split(","))
    unet_strides    = tuple(int(x) for x in str(cfg.get("unet_strides", "2,2,2,1")).split(","))

    # model
    model = FinetuneUNetClassificationModule(
        num_classes=K,
        lr=float(cfg.get("lr", 1e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-5)),
        pretrained_ckpt=pretrained_ckpt,
        class_names=class_names,
        freeze_encoder_epochs=int(cfg.get("freeze_encoder_epochs", 0)),
        linear_probe=bool(cfg.get("linear_probe", False)),
        init_mode=cfg.get("init", "pretrained"),
        in_channels=int(cfg.get("in_channels", 1)),
        unet_channels=unet_channels,
        unet_strides=unet_strides,
        unet_num_res_units=int(cfg.get("unet_num_res_units", 2)),
        unet_norm=cfg.get("unet_norm", "BATCH"),
        class_weights=class_weights,
        img_size=(96, 96, 96),
    )

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="finetune_cls_unet_best",
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
    infer_model = FinetuneUNetClassificationModule.load_from_checkpoint(best_ckpt, strict=False).to(device).eval()
    preds_csv = run_evaluation(infer_model, test_loader, class_names, device, preds_dir, tag="unet", wandb_logger=wandb_logger)

    del infer_model
    gc.collect()

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)


if __name__ == "__main__":
    main()

    