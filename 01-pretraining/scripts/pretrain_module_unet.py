# pretrain_module_unet.py - PyTorch Lightning module for pretraining with a UNet backbone (image+text or image-only)

# --- Setup ---

# imports
import numpy as np
import os
import random
import sys
# import seaborn as sns
# import umap
import wandb

import matplotlib.cm as cm
import matplotlib.pyplot as plt

from monai.networks.nets import Unet

from sklearn.preprocessing import normalize as sklearn_normalize

from transformers import AutoModel, AutoTokenizer

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

# get visualization functions from visualization_functions.py
sys.path.append("../../00-helpers/")
from visualization_functions import (
    log_embedding_umap_plot,
    log_images_batches_to_wandb_table
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- Lightweight decoder ---

# lightweight decoder head for pixel reconstruction from student features (for masked region reconstruction loss)
class LightDecoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


# --- Module ---

# main module class for pretraining using a UNet backbone (3D)
class PretrainModuleUnet(pl.LightningModule):

    # enable/disable umap logging
    ENABLE_UMAP_LOGGING = False

    # init
    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)  # save all hyperparameters to self.hparams

        # determine whether training with image+text or image-only based on config
        self.use_text = bool(config.get("model", {}).get("use_text", True))

        # base mask ratio to use after warmup
        self.base_mask_ratio = config["model"]["mask_ratio"]

        # parse downsampling config
        data_cfg = config["data"]
        down_cfg = data_cfg.get("downsample", {})

        # determine image size to use based on config options for cropping/downsampling
        if data_cfg.get("use_sub_patches", False):
            # use crops
            self.image_size = int(data_cfg.get("sub_patch_size", 64))
        elif bool(down_cfg.get("enabled", False)):
            # downsample
            self.image_size = int(down_cfg.get("target_size", data_cfg.get("base_patch_size", 96)))
        else:
            # use images without cropping/downsampling
            self.image_size = int(data_cfg.get("base_patch_size", 96))

        print(f"[INFO] Using {self.image_size}^3 size patches.", flush=True)

        self.mask_ratio = config["model"]["mask_ratio"]  # percentage of voxels to mask
        self.mask_ratio_warmup = config["model"]["mask_ratio_warmup"] # mask ratio to use during warmup epochs
        self.warmup_epochs = config["model"]["warmup_epochs"] # number of warmup epochs

        self.lr = config["model"]["lr"]  # learning rate
        self.mask_patch_size = config["model"]["mask_patch_size"]

        self.temp_student = config["model"]["temp_student"]  # temperature for student softmax
        self.temp_teacher = config["model"]["temp_teacher"]  # temperature for teacher softmax

        self.embed_dim = config["model"]["embed_dim"]

        # text-specific config options (only used if use_text=True)
        if self.use_text:
            self.text_model_name = config["model"]["text_model_name"]
            self.clip_temperature = config["model"]["clip_temperature"]
            init_temp = float(config["model"].get("clip_temperature", 0.07))
            # learnable logit scale for CLIP loss (instead of fixed temperature)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / init_temp), dtype=torch.float))
        # set text-related attributes to None for image-only runs
        else:
            self.text_model_name = None
            self.clip_temperature = None
            self.logit_scale = None

        # lightweight decoder head for pixel reconstruction from student features (for masked region reconstruction loss)
        self.reconstruction_head = LightDecoder(self.embed_dim)

        self.finetune_text = bool(config["model"].get("finetune_text", True)) if self.use_text else False
        self.text_top_k_layers = int(config["model"].get("text_top_k_layers", 4)) if self.use_text else 0
        self.text_finetune_start_epoch = int(config["model"].get("text_finetune_start_epoch", self.warmup_epochs)) if self.use_text else 10**9

        # track best metrics across the run
        self.best_metrics = {
            "train_loss": float("inf"),
            "val_loss": float("inf"),
            "train_clip_loss": float("inf"),
            "val_clip_loss": float("inf"),
            "train_ibot_loss": float("inf"),
            "val_ibot_loss": float("inf"),
        }

        # *** UNet Encoders ***
        unet_cfg = config["model"].get("unet", {})
        channels = tuple(unet_cfg.get("channels", (32, 64, 128, 256, 512)))
        strides = tuple(unet_cfg.get("strides", (2, 2, 2, 2)))
        num_res_units = int(unet_cfg.get("num_res_units", 2))
        norm = unet_cfg.get("norm", "INSTANCE")

        # student network
        self.student_encoder = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.embed_dim,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

        # ema teacher network (same architecture as student, updated using running average)
        self.teacher_encoder = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.embed_dim,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

        # initialize teacher with student weights
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_encoder.eval()

        self.register_buffer("ema_decay", torch.tensor(config["model"]["ema_decay"]))

        # text encoder (only used if use_text=True)
        if self.use_text:
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)
            self.text_encoder = AutoModel.from_pretrained(self.text_model_name)
            self.text_encoder.gradient_checkpointing_enable()

            # projection head to map text encoder output to same dimension as image encoder output for CLIP loss and alignment loss
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

            self._freeze_all_text()
            self.text_encoder.eval()
            
            # separate projection head for image features to be used in CLIP loss and alignment loss (instead of using student features directly) to allow more flexibility in learning a joint embedding space for images and text
            self.image_proj = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim),
            )

        # set text-related attributes to None for image-only runs
        else:
            self.text_tokenizer = None
            self.text_encoder = None
            self.text_proj = None
            self.image_proj = None

        # lists and count for logging to wandb
        self.train_batches_for_logging = []
        self.val_batches_for_logging = []
        self.train_log_count = 0
        self.val_log_count = 0
        self.max_log_images = config["model"]["max_log_images"]

        # loss weights
        self.distill_weight = config["loss_weights"]["distill_weight"]
        self.reconstruction_weight = config["loss_weights"]["reconstruction_weight"]
        self.align_weight = float(config["loss_weights"].get("align_weight", 0.0)) if self.use_text else 0.0
        self.clip_weight  = float(config["loss_weights"].get("clip_weight", 0.0)) if self.use_text else 0.0

        # fixed canonical report weights for comparability across runs
        report_weights_canon = config.get("report_weights", {})
        self.report_weights = {
            "clip": float(report_weights_canon.get("clip", 0.30)),
            "align": float(report_weights_canon.get("align", 0.10)),
            "recon": float(report_weights_canon.get("recon", 0.30)),
            "distill": float(report_weights_canon.get("distill", 0.30)),
        }
        if not self.use_text:
            # ensure report metric is meaningful and comparable for image-only runs
            self.report_weights["clip"] = 0.0
            self.report_weights["align"] = 0.0

    # *** Text Freezing/Unfreezing Helpers ***

    # helper function to freeze all text encoder layers (used at init and when unfreezing top k layers)
    def _freeze_all_text(self):
        if not self.use_text or self.text_encoder is None:
            return
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    # helper function to unfreeze top k layers of text encoder (called at start of each epoch based on config options)
    def _unfreeze_top_k_text_layers(self, k: int):
        if not self.use_text or self.text_encoder is None:
            return
        # freeze everything first
        self._freeze_all_text()

        num_layers = self.text_encoder.config.num_hidden_layers
        start = max(0, num_layers - int(k))

        for name, p in self.text_encoder.named_parameters():
            if any(name.startswith(f"encoder.layer.{i}.") for i in range(start, num_layers)):
                p.requires_grad = True

            # unfreeze embeddings and layer norms too
            if name.startswith(("embeddings.",)) or ".LayerNorm." in name:
                p.requires_grad = True

        if hasattr(self.text_encoder, "pooler"):
            for p in self.text_encoder.pooler.parameters():
                p.requires_grad = True

    # *** Mask Generation Helper ***

    # creates patch-level binary mask for each volume in batch
    def generate_patch_mask(self, x: torch.Tensor, base_mask_patch_size: int) -> torch.Tensor:

        B, C, D, H, W = x.shape
        mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=x.device)

        for b in range(B):
            # randomly select patch size (base or double)
            if torch.rand((), device=x.device) < 0.5:
                mask_patch_size = base_mask_patch_size
            else:
                mask_patch_size = base_mask_patch_size * 2

            d_mask_patches = D // mask_patch_size
            h_mask_patches = H // mask_patch_size
            w_mask_patches = W // mask_patch_size
            num_total_mask_patches = d_mask_patches * h_mask_patches * w_mask_patches

            num_masked_mask_patches = max(1, int(self.mask_ratio * num_total_mask_patches))

            mask_patch_indices = torch.randperm(num_total_mask_patches, device=x.device)[:num_masked_mask_patches]

            for idx in mask_patch_indices:
                z = (idx // (h_mask_patches * w_mask_patches)) * mask_patch_size
                y = ((idx % (h_mask_patches * w_mask_patches)) // w_mask_patches) * mask_patch_size
                x_ = (idx % w_mask_patches) * mask_patch_size
                mask[b, :, z : z + mask_patch_size, y : y + mask_patch_size, x_ : x_ + mask_patch_size] = True

        return mask

    # *** Encoding Helpers ***

    # function to encode text prompts into embeddings using text encoder and projection head (only used if use_text=True)
    def encode_text(self, texts):
        if not self.use_text:
            raise RuntimeError("encode_text called but model.use_text is False")

        # tokenize and encode text prompts
        tokens = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        # only build grads for text when it's trainable
        with torch.set_grad_enabled(self.text_encoder.training):
            out = self.text_encoder(**tokens)
            pooled = out.last_hidden_state.mean(dim=1)
            proj = self.text_proj(pooled)
            return F.normalize(proj, dim=-1)

    # function to encode images into embeddings using student encoder and projection head (projection head only used if use_text=True, i.e. when computing CLIP loss and alignment loss)
    def encode_image(self, x: torch.Tensor, mask: torch.Tensor, network: nn.Module):
        x_masked = x.clone()
        x_masked[mask] = 0
        features = network(x_masked)
        features_norm = F.normalize(features, dim=1)
        return features, features_norm

    # *** Losses ***

    # function to compute CLIP loss between image and text embeddings
    def compute_clip_loss(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor):
        if not self.use_text:
            return torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)
        logits = (image_embeds @ text_embeds.T) * logit_scale
        targets = torch.arange(logits.shape[0], device=self.device)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2
        return loss, logit_scale

    def forward(self, x: torch.Tensor):
        return self.student_encoder(x)

    # function to compute alignment loss between image and text embeddings
    def compute_alignment_loss(self, image_embeds, text_embeds, detach_text: bool = True):
        if not self.use_text:
            return torch.tensor(0.0, device=self.device)
        if detach_text:
            text_embeds = text_embeds.detach()
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return 1 - F.cosine_similarity(image_embeds, text_embeds, dim=-1).mean()

    # function to compute iBOT loss (distillation loss + reconstruction loss + alignment loss)
    def compute_ibot_loss(
        self,
        student_out_norm: torch.Tensor,
        teacher_out_norm: torch.Tensor,
        x: torch.Tensor,
        mask: torch.Tensor,
        student_feats_raw: torch.Tensor,
        student_image_embed: torch.Tensor,
        text_embed: torch.Tensor,
    ):
        B, C, D, H, W = student_out_norm.shape

        mask_downsampled = F.interpolate(mask.float(), size=(D, H, W), mode="nearest")
        x_downsampled = F.interpolate(x, size=(D, H, W), mode="trilinear", align_corners=False)

        N = D * H * W
        stu_tok = student_out_norm.view(B, C, N)
        tea_tok = teacher_out_norm.view(B, C, N).detach()
        mask_bool = mask_downsampled.view(B, 1, N).to(torch.bool).squeeze(1)

        # compute distillation loss (KL divergence between student and teacher features on masked regions only)
        if mask_bool.any():
            student_selection = stu_tok.transpose(1, 2)[mask_bool].float()
            teacher_selection = tea_tok.transpose(1, 2)[mask_bool].float()

            student_logprobs = F.log_softmax(student_selection / self.temp_student, dim=-1)
            teacher_probs = F.softmax(teacher_selection / self.temp_teacher, dim=-1)

            distill_loss = F.kl_div(student_logprobs, teacher_probs, reduction="batchmean")

        # if no masked tokens (can happen with small images and large patch sizes), set distillation loss to zero
        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        # reconstruction head decodes from raw (unnormalized) features
        student_recon = self.reconstruction_head(student_feats_raw)

        # pixel reconstruction loss (L1) on masked regions only
        recon_loss_l1 = (
            F.l1_loss(student_recon * mask_downsampled, x_downsampled * mask_downsampled, reduction="sum")
            / (mask_downsampled.sum() + 1e-8)
        )

        # alignment loss between student image embeddings and text embeddings
        alignment_loss = self.compute_alignment_loss(student_image_embed, text_embed)

        # combine losses with weights to get total iBOT loss
        total_ibot_loss = (
            self.distill_weight * distill_loss
            + self.reconstruction_weight * recon_loss_l1
            + self.align_weight * alignment_loss
        )

        # create composite reconstruction for visualization by combining reconstructed masked regions with original unmasked regions
        recon_composite = x_downsampled * (1.0 - mask_downsampled) + student_recon * mask_downsampled

        # return all losses and recon composite for logging and optimization
        return total_ibot_loss, recon_composite, distill_loss, recon_loss_l1, alignment_loss

    # *** Logging Helpers ***

    def log_image_and_embeddings(
        self,
        x,
        mask,
        student_out,
        texts,
        is_train: bool = True,
        student_image_embed=None,
        text_embed=None,
        student_recon=None,
    ):

        # if student_image_embed or text_embed are not provided, compute them using the projection heads (only used for CLIP loss and alignment loss when use_text=True)
        if student_recon is None:
            student_recon = self.reconstruction_head(student_out)

        # create composite reconstruction for visualization by combining reconstructed masked regions with original unmasked regions
        recon_image = student_recon.clamp(0, 1)

        # create masked input for visualization by applying mask to original image
        masked_input = x.clone()
        masked_input[mask] = 0

        # get count of images for logging
        log_count = self.train_log_count if is_train else self.val_log_count
        batches_for_logging = self.train_batches_for_logging if is_train else self.val_batches_for_logging

        # if still have capacity to log more images, add current batch (or portion of it) to list of batches to log
        if log_count < self.max_log_images:
            num_to_add = min(self.max_log_images - log_count, x.shape[0])
            batches_for_logging.append(
                (
                    x[:num_to_add].detach().to(torch.float32).cpu(),
                    masked_input[:num_to_add].detach().to(torch.float32).cpu(),
                    recon_image[:num_to_add].detach().to(torch.float32).cpu(),
                )
            )
            if is_train:
                self.train_log_count += num_to_add
            else:
                self.val_log_count += num_to_add

        embed_dict_attr = "train_embeddings" if is_train else "val_embeddings"
        embed_dict = getattr(self, embed_dict_attr, None)
        if embed_dict is None:
            embed_dict = {"image": [], "text": [], "label": []}
            setattr(self, embed_dict_attr, embed_dict)

        # if student_image_embed or text_embed are not provided, compute them using the projection heads (only used for CLIP loss and alignment loss when use_text=True)
        if student_image_embed is None:

            # for image embeddings, apply projection head to normalized student features if using text (to get CLIP and alignment loss), otherwise just use normalized student features directly
            if self.image_proj is not None:
                student_image_embed = F.normalize(self.image_proj(student_out), dim=-1).detach().cpu()
            else:
                student_image_embed = student_out.detach().cpu()

        else:
            student_image_embed = student_image_embed.detach().cpu()

        if self.use_text:
            if text_embed is None:
                text_embed = self.encode_text(texts).detach().cpu()
            else:
                text_embed = text_embed.detach().cpu()

            # cache embeddings and labels for UMAP visualization at the end of the epoch
            embed_dict["image"].append(student_image_embed)
            embed_dict["text"].append(text_embed)
            embed_dict["label"].extend(texts)

    # *** Lightning Hooks ***

    # on train start, print debug info about rank and world size for distributed training
    def on_train_start(self):
        print(f"[DEBUG] RANK: {self.global_rank}, world_size={self.trainer.world_size}", flush=True)

    # on epoch start, update mask ratio based on warmup schedule and freeze/unfreeze text encoder layers based on config options
    def on_train_epoch_start(self):
        if self.current_epoch < self.warmup_epochs:
            self.mask_ratio = self.mask_ratio_warmup

            if self.use_text:
                self._freeze_all_text()
                self.text_encoder.eval()
        else:
            self.mask_ratio = self.base_mask_ratio
            if self.use_text and self.finetune_text and self.current_epoch >= self.text_finetune_start_epoch:
                self._unfreeze_top_k_text_layers(self.text_top_k_layers)
                self.text_encoder.train()
                try:
                    self.text_encoder.gradient_checkpointing_disable()
                except AttributeError:
                    pass
            else:
                if self.use_text:
                    self._freeze_all_text()
                    self.text_encoder.eval()

    # shared step function for training and validation to avoid code duplication
    def shared_step(self, batch, batch_idx, is_train: bool = True):

        # get images and texts from batch (texts will be None if image-only dataset)
        x = batch["image"]
        texts = batch.get("text", None)

        if self.use_text and texts is None:
            raise RuntimeError("use_text=True but batch has no 'text' field. Check datamodule/dataset.")

        # squeeze stray trailing singleton dimension if present: (B,C,D,H,W,1) -> (B,C,D,H,W)
        if x.ndim == 6 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # ensure channel-first (B,1,D,H,W); if given (B,D,H,W,1) then permute
        if x.ndim == 5 and x.shape[1] != 1 and x.shape[-1] == 1:
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        # add channel dimension if missing
        if x.ndim == 4:
            x = x.unsqueeze(1)

        # generate random patch mask for each image in batch
        mask = self.generate_patch_mask(x, self.mask_patch_size)

        # encode images with student and teacher encoders to get features and normalized features for loss computation
        student_feats_raw, student_out_norm = self.encode_image(x, mask, self.student_encoder)
        with torch.no_grad():
            _, teacher_out_norm = self.encode_image(x, torch.zeros_like(mask, dtype=torch.bool), self.teacher_encoder)

        # compute image and text embeddings for CLIP loss and alignment loss (only if use_text=True, otherwise set to None)
        if self.use_text:
            student_image_embed = F.normalize(self.image_proj(student_out_norm), dim=-1)
            text_embed = self.encode_text(texts)
            clip_loss, logit_scale = self.compute_clip_loss(student_image_embed, text_embed)
        else:
            student_image_embed, text_embed = None, None
            clip_loss, logit_scale = torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)

        # compute iBOT loss (distillation loss + reconstruction loss + alignment loss)
        ibot_loss, recon_composite, distill_loss, recon_loss_l1, align_loss = self.compute_ibot_loss(
            student_out_norm,
            teacher_out_norm,
            x,
            mask,
            student_feats_raw,
            student_image_embed,
            text_embed,
        )

        # combine iBOT loss and CLIP loss with weight to get total loss for optimization
        total_loss = ibot_loss + self.clip_weight * clip_loss

        # log losses and metrics
        log_prefix = "train" if is_train else "val"
        self.log(
            f"{log_prefix}_loss",
            total_loss,
            on_step=is_train,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_clip_loss",
            clip_loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_ibot_loss",
            ibot_loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_distill_loss",
            distill_loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_recon_loss_l1",
            recon_loss_l1,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_align_loss",
            align_loss,
            on_step=True,
            on_epoch=True,
            batch_size=x.shape[0],
            sync_dist=True,
        )
        self.log(
            f"{log_prefix}_logit_scale",
            logit_scale,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # log composite loss for easier comparability across runs with different loss weights (only log at epoch level to reduce noise)
        if not is_train:
            val_loss_report = (
                self.report_weights["clip"] * clip_loss
                + self.report_weights["align"] * align_loss
                + self.report_weights["recon"] * recon_loss_l1
                + self.report_weights["distill"] * distill_loss
            )
            self.log(
                "val_loss_report",
                val_loss_report,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=x.shape[0],
            )

        # composite reconstruction for logging: resize to input size for logging
        recon_for_log = F.interpolate(recon_composite, size=x.shape[2:], mode="trilinear", align_corners=False)
        self.log_image_and_embeddings(
            x,
            mask,
            student_out_norm,
            texts if self.use_text else None,
            is_train=is_train,
            student_image_embed=student_image_embed,
            text_embed=text_embed,
            student_recon=recon_for_log,
        )

        return total_loss

    # separate training and validation step functions that call shared_step with appropriate flag to avoid code duplication
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, is_train=True)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, is_train=False)

    # at the end of each epoch, log images and embeddings to wandb and update best metrics
    def on_train_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        if self.train_batches_for_logging:
            log_images_batches_to_wandb_table(
                logger=self.logger,
                batches=self.train_batches_for_logging,
                prefix="Train",
                step_or_epoch=self.current_epoch,
                max_rows=self.max_log_images,
                log_histograms=False,
            )
        self.train_batches_for_logging.clear()
        self.train_log_count = 0

        if self.ENABLE_UMAP_LOGGING and hasattr(self, "train_embeddings") and self.train_embeddings:
            if self.current_epoch % 5 == 0 or self.current_epoch == 0:
                log_embedding_umap_plot(
                    logger=self.logger,
                    embeddings_dict=self.train_embeddings,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    tag="Train",
                )
            del self.train_embeddings

    # at the end of each epoch, log images and embeddings to wandb and update best metrics
    def on_validation_epoch_end(self):
        if not self.trainer.is_global_zero:
            return

        if self.val_batches_for_logging:
            log_images_batches_to_wandb_table(
                logger=self.logger,
                batches=self.val_batches_for_logging,
                prefix="Val",
                step_or_epoch=self.current_epoch,
                max_rows=self.max_log_images,
                log_histograms=False,
            )
        self.val_batches_for_logging.clear()
        self.val_log_count = 0

        if self.ENABLE_UMAP_LOGGING and hasattr(self, "val_embeddings"):
            if self.current_epoch % 5 == 0 or self.current_epoch == 0:
                log_embedding_umap_plot(
                    logger=self.logger,
                    embeddings_dict=self.val_embeddings,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    tag="Val",
                )
            del self.val_embeddings

        c = self.trainer.callback_metrics
        self.update_best_metrics("train_loss", c.get("train_loss"))
        self.update_best_metrics("train_clip_loss", c.get("train_clip_loss"))
        self.update_best_metrics("train_ibot_loss", c.get("train_ibot_loss"))
        self.update_best_metrics("val_loss", c.get("val_loss"))
        self.update_best_metrics("val_clip_loss", c.get("val_clip_loss"))
        self.update_best_metrics("val_ibot_loss", c.get("val_ibot_loss"))

    # at the end of each training batch, update ema teacher parameters from student parameters using the specified decay rate
    def on_train_batch_end(self, outputs, batch, batch_idx):
        # update ema teacher parameters from student (no grad)
        with torch.no_grad():
            m = float(self.ema_decay.item() if isinstance(self.ema_decay, torch.Tensor) else self.ema_decay)
            for student_param, teacher_param in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
                teacher_param.mul_(m).add_(student_param, alpha=1 - m)

    def on_fit_end(self):
        if not self.trainer.is_global_zero:
            return

        if hasattr(self, "logger") and hasattr(self.logger, "experiment"):
            exp = self.logger.experiment
            table = wandb.Table(
                columns=[
                    "best_train_loss",
                    "best_val_loss",
                    "best_train_ibot_loss",
                    "best_val_ibot_loss",
                ]
            )
            if self.use_text:
                table = wandb.Table(columns=[
                    "best_train_loss",
                    "best_val_loss",
                    "best_train_clip_loss",
                    "best_val_clip_loss",
                    "best_train_ibot_loss",
                    "best_val_ibot_loss",
                ])
                table.add_data(
                    self.best_metrics["train_loss"],
                    self.best_metrics["val_loss"],
                    self.best_metrics["train_clip_loss"],
                    self.best_metrics["val_clip_loss"],
                    self.best_metrics["train_ibot_loss"],
                    self.best_metrics["val_ibot_loss"],
                )
            else:
                table.add_data(
                    self.best_metrics["train_loss"],
                    self.best_metrics["val_loss"],
                    self.best_metrics["train_ibot_loss"],
                    self.best_metrics["val_ibot_loss"],
                )

            exp.log({"best_metrics_table": table})
            for k, v in self.best_metrics.items():
                exp.summary[f"best_{k}"] = v

    def configure_optimizers(self):
        # skip frozen teacher params
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=self.lr)

    # helper function to update best metrics at the end of each epoch
    def update_best_metrics(self, name, value):
        if value is None:
            return
        try:
            v = float(value.detach().cpu().item() if hasattr(value, "detach") else value)
        except Exception:
            return
        if v < self.best_metrics[name]:
            self.best_metrics[name] = v
