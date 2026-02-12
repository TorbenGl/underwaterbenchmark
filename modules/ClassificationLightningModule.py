"""
Generic Lightning Module for Image Classification.

This module can work with any classification model that returns logits and loss.
Mirrors the structure of SemanticLightningModule but adapted for classification.
"""

import torch
import torch.nn as nn
import lightning as L
from torchmetrics import Accuracy, F1Score
import wandb
from typing import Optional, Dict, Any


class ClassificationLightningModule(L.LightningModule):
    """
    Generic Lightning module for image classification.

    Works with any model that has:
    - forward(pixel_values, labels) -> dict with 'logits' and 'loss'

    Args:
        model: Classification model (e.g., ClassificationModel)
        num_classes: Number of classification classes
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimizer
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

        # Metrics
        metric_kwargs = {
            "task": "multiclass",
            "num_classes": num_classes,
        }

        # Accuracy
        self.train_acc = Accuracy(**metric_kwargs)
        self.val_acc = Accuracy(**metric_kwargs)
        self.test_acc = Accuracy(**metric_kwargs)

        # F1 Score (macro-averaged)
        self.train_f1 = F1Score(**metric_kwargs, average="macro")
        self.val_f1 = F1Score(**metric_kwargs, average="macro")
        self.test_f1 = F1Score(**metric_kwargs, average="macro")

        # Per-class F1 for detailed logging
        self.val_f1_per_class = F1Score(**metric_kwargs, average=None)
        self.test_f1_per_class = F1Score(**metric_kwargs, average=None)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Filter out frozen backbone weights when saving checkpoints."""
        if self.hparams.freeze_backbone:
            state_dict = checkpoint["state_dict"]
            keys_to_remove = [k for k in state_dict.keys() if "backbone" in k]
            for key in keys_to_remove:
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Handle loading checkpoints that don't have backbone weights."""
        state_dict = checkpoint["state_dict"]
        has_backbone = any("backbone" in k for k in state_dict.keys())

        if not has_backbone:
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

        for name, param in self.model.named_parameters():
            num_params = param.numel()
            is_trainable = param.requires_grad

            if "backbone" in name:
                backbone_params += num_params
                if is_trainable:
                    backbone_trainable += num_params
            else:
                head_params += num_params
                if is_trainable:
                    head_trainable += num_params

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

    def _apply_augmentation(self, images: torch.Tensor, augmentation: nn.Module) -> torch.Tensor:
        """Apply augmentation to images only (classification has no masks).

        Reuses the existing Kornia segmentation augmentations by passing a
        dummy mask and discarding the returned mask.
        """
        dummy_mask = torch.zeros(
            images.shape[0], images.shape[2], images.shape[3],
            dtype=torch.long, device=images.device,
        )
        images, _ = augmentation(images, dummy_mask)
        return images

    def _step(self, batch: Dict[str, torch.Tensor], stage: str = "train"):
        """Common step for train/val/test."""
        images = batch["pixel_values"]
        labels = batch["labels"]

        # Apply augmentations (image-only)
        if stage == "train" and self.train_augmentation is not None:
            images = self._apply_augmentation(images, self.train_augmentation)
        elif stage in ["val", "test"] and self.val_augmentation is not None:
            images = self._apply_augmentation(images, self.val_augmentation)

        # Forward pass
        outputs = self(pixel_values=images, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        preds = logits.argmax(dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, stage="train")

        self.train_acc(preds, labels)
        self.train_f1(preds, labels)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, stage="val")

        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.val_f1_per_class(preds, labels)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        """Log per-class F1 scores."""
        per_class_f1 = self.val_f1_per_class.compute()
        if per_class_f1.ndim > 0:
            for i, f1 in enumerate(per_class_f1):
                if not torch.isnan(f1):
                    label = self.id2label.get(i, f"class_{i}")
                    self.log(f"val/f1_{label}", f1, sync_dist=True)
        self.val_f1_per_class.reset()

        # Log confusion matrix to wandb
        self._log_confusion_matrix()

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch, stage="test")

        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_f1_per_class(preds, labels)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)

        return loss

    def on_test_epoch_end(self):
        """Log per-class F1 scores."""
        per_class_f1 = self.test_f1_per_class.compute()
        if per_class_f1.ndim > 0:
            for i, f1 in enumerate(per_class_f1):
                if not torch.isnan(f1):
                    label = self.id2label.get(i, f"class_{i}")
                    self.log(f"test/f1_{label}", f1, sync_dist=True)
        self.test_f1_per_class.reset()

    def _log_confusion_matrix(self):
        """Log confusion matrix to wandb."""
        if not isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            return

        try:
            class_names = [
                self.id2label.get(i, f"class_{i}")
                for i in range(self.num_classes)
            ]
            self.logger.experiment.log({
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    y_true=None,
                    preds=None,
                    class_names=class_names,
                )
            })
        except Exception:
            pass

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
