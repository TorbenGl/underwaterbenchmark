"""
Generic Lightning Module for Semantic Segmentation.

This module can work with any segmentation model that returns logits and loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torchmetrics import JaccardIndex, Accuracy
import wandb
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


class SemanticLightningModule(L.LightningModule):
    """
    Generic Lightning module for semantic segmentation.

    Works with any model that has:
    - forward(pixel_values, labels) -> dict with 'logits' and 'loss'

    Args:
        model: Segmentation model (e.g., UperNetSegmentationModel)
        num_classes: Number of segmentation classes
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimizer
        loss_ignore_index: Index to ignore in metrics (e.g., 255 for padding pixels).
            Pixels with this value are excluded from all metric computations.
        metrics_ignore_index: Index to exclude from mean IoU/Acc (e.g., 0 for background).
            Per-class metrics are still logged for ALL classes including the ignored one.
        lr_scheduler: Scheduler type ("cosine", "step", "polynomial", None)
        warmup_epochs: Number of warmup epochs
        max_epochs: Maximum training epochs
        freeze_backbone: Whether to freeze backbone weights
        backbone_lr_factor: Learning rate multiplier for backbone (default 0.1)
        train_augmentation: Kornia augmentation module for training
        val_augmentation: Kornia augmentation module for validation
        id2label: Optional mapping from class IDs to labels for logging
    """

    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_ignore_index: Optional[int] = 255,
        metrics_ignore_index: Optional[int] = None,
        lr_scheduler: Optional[str] = "cosine",
        warmup_epochs: int = 0,
        max_epochs: int = 100,
        freeze_backbone: bool = True,
        backbone_lr_factor: float = 0.1,
        train_augmentation: Optional[nn.Module] = None,
        val_augmentation: Optional[nn.Module] = None,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "train_augmentation", "val_augmentation"])
        self.model = model
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.id2label = id2label or {}

        # Augmentation modules
        self.train_augmentation = train_augmentation
        self.val_augmentation = val_augmentation

        # Freeze backbone if specified
        if freeze_backbone and hasattr(model, 'freeze_backbone'):
            model.freeze_backbone()
            print("Backbone frozen")

        # Store ignore indices
        self.loss_ignore_index = loss_ignore_index
        self.metrics_ignore_index = metrics_ignore_index

        # Base metric kwargs with loss_ignore_index to handle padding (255)
        # This ensures validation passes when targets contain padding values
        base_metric_kwargs = {
            "task": "multiclass",
            "num_classes": num_classes,
        }
        if loss_ignore_index is not None:
            base_metric_kwargs["ignore_index"] = loss_ignore_index

        # Per-class IoU metrics - uses loss_ignore_index to skip padding pixels
        # Reports IoU for ALL classes including background
        self.train_iou_per_class = JaccardIndex(**base_metric_kwargs, average=None)
        self.val_iou_per_class = JaccardIndex(**base_metric_kwargs, average=None)
        self.test_iou_per_class = JaccardIndex(**base_metric_kwargs, average=None)

        # Accuracy metrics - uses loss_ignore_index to skip padding pixels
        self.train_acc = Accuracy(**base_metric_kwargs)
        self.val_acc = Accuracy(**base_metric_kwargs)
        self.test_acc = Accuracy(**base_metric_kwargs)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Filter out frozen backbone weights when saving checkpoints."""
        if self.hparams.freeze_backbone:
            state_dict = checkpoint["state_dict"]
            # Remove all backbone parameters
            keys_to_remove = [k for k in state_dict.keys() if "backbone" in k]
            for key in keys_to_remove:
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle loading checkpoints that don't have backbone weights."""
        state_dict = checkpoint["state_dict"]

        # Check if backbone weights are missing
        has_backbone = any("backbone" in k for k in state_dict.keys())

        if not has_backbone:
            # Add current (pretrained) backbone weights to checkpoint
            for name, param in self.model.named_parameters():
                if "backbone" in name:
                    state_dict[f"model.{name}"] = param.data

    def on_fit_start(self):
        """Log parameter counts for backbone and head separately."""
        self._print_parameter_summary()

    def _print_parameter_summary(self):
        """Print detailed parameter summary separated by backbone and head."""
        backbone_params = 0
        backbone_trainable = 0
        head_params = 0
        head_trainable = 0

        backbone_details = []
        head_details = []

        for name, param in self.model.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad

            if "backbone" in name:
                backbone_params += num_params
                if is_trainable:
                    backbone_trainable += num_params
                backbone_details.append((name, num_params, is_trainable))
            else:
                head_params += num_params
                if is_trainable:
                    head_trainable += num_params
                head_details.append((name, num_params, is_trainable))

        total_params = backbone_params + head_params
        total_trainable = backbone_trainable + head_trainable

        print("\n" + "=" * 70)
        print("MODEL PARAMETER SUMMARY")
        print("=" * 70)

        print(f"\n{'BACKBONE PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Total:     {backbone_params:>15,} ({backbone_params / 1e6:.2f}M)")
        print(f"  Trainable: {backbone_trainable:>15,} ({backbone_trainable / 1e6:.2f}M)")
        print(f"  Frozen:    {backbone_params - backbone_trainable:>15,} ({(backbone_params - backbone_trainable) / 1e6:.2f}M)")

        print(f"\n{'HEAD PARAMETERS':^70}")
        print("-" * 70)
        print(f"  Total:     {head_params:>15,} ({head_params / 1e6:.2f}M)")
        print(f"  Trainable: {head_trainable:>15,} ({head_trainable / 1e6:.2f}M)")
        print(f"  Frozen:    {head_params - head_trainable:>15,} ({(head_params - head_trainable) / 1e6:.2f}M)")

        print(f"\n{'TOTAL':^70}")
        print("-" * 70)
        print(f"  Total:     {total_params:>15,} ({total_params / 1e6:.2f}M)")
        print(f"  Trainable: {total_trainable:>15,} ({total_trainable / 1e6:.2f}M)")
        print(f"  Frozen:    {total_params - total_trainable:>15,} ({(total_params - total_trainable) / 1e6:.2f}M)")
        print("=" * 70 + "\n")

        # Log to wandb if available
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self.logger.experiment.config.update({
                "params/backbone_total": backbone_params,
                "params/backbone_trainable": backbone_trainable,
                "params/head_total": head_params,
                "params/head_trainable": head_trainable,
                "params/total": total_params,
                "params/trainable": total_trainable,
            }, allow_val_change=True)

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None):
        return self.model(pixel_values=pixel_values, labels=labels)

    def _step(self, batch: Dict[str, torch.Tensor], stage: str = "train"):
        """Common step for train/val/test."""
        images = batch["pixel_values"]
        labels = batch["labels"]

        # Apply augmentations
        if stage == "train" and self.train_augmentation is not None:
            images, labels = self.train_augmentation(images, labels)
        elif stage in ["val", "test"] and self.val_augmentation is not None:
            images, labels = self.val_augmentation(images, labels)

        # Forward pass
        outputs = self(pixel_values=images, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        # Upsample logits if needed
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = F.interpolate(
                logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )

        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def _compute_mean_iou(self, per_class_iou: torch.Tensor) -> torch.Tensor:
        """Compute mean IoU, excluding metrics_ignore_index class if set."""
        valid_mask = ~torch.isnan(per_class_iou)
        if self.metrics_ignore_index is not None and self.metrics_ignore_index < len(per_class_iou):
            valid_mask[self.metrics_ignore_index] = False
        valid_ious = per_class_iou[valid_mask]
        return valid_ious.mean() if len(valid_ious) > 0 else torch.tensor(0.0)

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, stage="train")

        self.train_iou_per_class(preds, labels)
        self.train_acc(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def on_train_epoch_end(self):
        """Log mean IoU excluding background."""
        per_class_iou = self.train_iou_per_class.compute()
        if per_class_iou.ndim > 0:
            mean_iou = self._compute_mean_iou(per_class_iou)
            self.log("train/iou", mean_iou, sync_dist=True, prog_bar=True)
        self.train_iou_per_class.reset()

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, stage="val")

        self.val_iou_per_class(preds, labels)
        self.val_acc(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)

        if batch_idx == 0:
            self._log_predictions(batch, preds, labels)

        return loss

    def on_validation_epoch_end(self):
        """Log per-class IoU and mean IoU excluding background."""
        per_class_iou = self.val_iou_per_class.compute()
        if per_class_iou.ndim > 0:
            # Log mean IoU (excluding metrics_ignore_index)
            mean_iou = self._compute_mean_iou(per_class_iou)
            self.log("val/iou", mean_iou, sync_dist=True, prog_bar=True)
            # Log per-class IoU
            for i, iou in enumerate(per_class_iou):
                if not torch.isnan(iou):
                    label = self.id2label.get(i, f"class_{i}")
                    self.log(f"val/iou_{label}", iou, sync_dist=True)
        self.val_iou_per_class.reset()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, stage="test")

        self.test_iou_per_class(preds, labels)
        self.test_acc(preds, labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """Log per-class IoU and mean IoU excluding background."""
        per_class_iou = self.test_iou_per_class.compute()
        if per_class_iou.ndim > 0:
            # Log mean IoU (excluding metrics_ignore_index)
            mean_iou = self._compute_mean_iou(per_class_iou)
            self.log("test/iou", mean_iou, sync_dist=True)
            # Log per-class IoU
            for i, iou in enumerate(per_class_iou):
                if not torch.isnan(iou):
                    label = self.id2label.get(i, f"class_{i}")
                    self.log(f"test/iou_{label}", iou, sync_dist=True)
        self.test_iou_per_class.reset()

    def _log_predictions(self, batch, preds, labels, num_samples: int = 4):
        """Log prediction visualizations to wandb as side-by-side images."""
        if not isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            return

        images = batch["pixel_values"][:num_samples]
        preds = preds[:num_samples]
        labels = labels[:num_samples]

        # Create a colormap for the classes
        cmap = plt.cm.get_cmap("tab20", self.num_classes)

        wandb_images = []
        for img, pred, label in zip(images, preds, labels):
            # Denormalize image (ImageNet normalization)
            img_np = img.cpu().permute(1, 2, 0).numpy()
            img_np = (img_np * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]
            img_np = img_np.clip(0, 1)

            pred_mask = pred.cpu().numpy()
            label_mask = label.cpu().numpy()

            # Replace loss_ignore_index with metrics_ignore_index for visualization
            if self.loss_ignore_index is not None and self.metrics_ignore_index is not None:
                label_mask = np.where(label_mask == self.loss_ignore_index, self.metrics_ignore_index, label_mask)

            # Create colored masks using colormap
            pred_colored = cmap(pred_mask / max(self.num_classes - 1, 1))[:, :, :3]
            label_colored = cmap(label_mask / max(self.num_classes - 1, 1))[:, :, :3]

            # Create side-by-side figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            axes[0].imshow(img_np)
            axes[0].set_title("Input Image")
            axes[0].axis("off")

            axes[1].imshow(label_colored)
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(pred_colored)
            axes[2].set_title("Prediction")
            axes[2].axis("off")

            plt.tight_layout()

            # Convert figure to image
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            fig_array = np.asarray(buf)[:, :, :3]  # Drop alpha channel

            wandb_images.append(wandb.Image(fig_array))
            plt.close(fig)

        self.logger.experiment.log({"val/predictions": wandb_images})

    def configure_optimizers(self):
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

        # Build parameter groups
        param_groups = [{"params": head_params, "lr": self.learning_rate}]

        if backbone_params:
            backbone_lr = self.learning_rate * self.hparams.backbone_lr_factor
            param_groups.insert(0, {"params": backbone_params, "lr": backbone_lr})

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # Learning rate scheduler
        scheduler_type = self.hparams.lr_scheduler
        if not scheduler_type:
            return optimizer

        max_epochs = self.hparams.max_epochs
        warmup_epochs = self.hparams.warmup_epochs

        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
            )
        elif scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        elif scheduler_type == "polynomial":
            scheduler = torch.optim.lr_scheduler.PolynomialLR(
                optimizer, total_iters=max_epochs, power=0.9
            )
        else:
            return optimizer

        # Add warmup
        if warmup_epochs > 0:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_scheduler = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }
