"""
Classification Model.

This module provides the classification model that combines a backbone adapter
with a linear classification head. Same interface pattern as the segmentation
models: forward(pixel_values, labels) -> {"logits": ..., "loss": ...}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from models.upernet_model import pad_to_patch_size


class ClassificationModel(nn.Module):
    """
    Classification model with a configurable backbone.

    Combines:
    - A ClassificationBackboneAdapter for feature extraction → [B, hidden_size]
    - A linear classification head → [B, num_classes]

    Args:
        backbone: The backbone adapter (ClassificationBackboneAdapter)
        num_classes: Number of classification classes
        hidden_size: Hidden size from the backbone
        auto_pad: Whether to auto-pad inputs to be divisible by the backbone patch size
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden_size: int,
        auto_pad: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.auto_pad = auto_pad
        self.patch_size = getattr(backbone, "patch_size", 14)

        self.backbone = backbone
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            pixel_values: Input images [B, C, H, W]
            labels: Ground truth class labels [B] (long tensor)

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        # Auto-pad to patch size if enabled
        if self.auto_pad:
            pixel_values, _, _ = pad_to_patch_size(
                pixel_values=pixel_values,
                labels=None,
                patch_size=self.patch_size,
            )

        features = self.backbone(pixel_values)  # [B, hidden_size]
        logits = self.classifier(features)  # [B, num_classes]

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            result["loss"] = loss

        return result

    def get_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get classification logits without computing loss."""
        return self.forward(pixel_values)["logits"]

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
