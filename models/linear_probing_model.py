"""
Linear Probing Semantic Segmentation Model.

This module provides a linear probing segmentation model that freezes the backbone
and trains only a single 1x1 convolution (linear classifier) on top of backbone
features. Same signature as UperNetSegmentationModel for drop-in replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple

from models.upernet_model import pad_to_patch_size


class LinearProbingSegmentationModel(nn.Module):
    """
    Linear probing model for semantic segmentation with a frozen backbone.

    Freezes the backbone and trains a single 1x1 convolution (linear classifier)
    on the backbone's last feature map. Loss is computed with cross-entropy
    ignoring the specified ignore index.

    Args:
        backbone: The backbone module (e.g., UperNetBackboneAdapter)
        num_classes: Number of segmentation classes
        hidden_size: Hidden size (output channels of the backbone feature maps)
        hidden_sizes: Unused, kept for signature compatibility with UperNetSegmentationModel
        use_auxiliary_head: Unused, kept for signature compatibility
        loss_ignore_index: Index to ignore in loss computation
        auto_pad: Whether to auto-pad inputs to be divisible by the backbone patch size
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        hidden_size: int = 512,
        hidden_sizes: Optional[List[int]] = None,
        use_auxiliary_head: bool = False,
        loss_ignore_index: int = 255,
        auto_pad: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.loss_ignore_index = loss_ignore_index
        self.auto_pad = auto_pad

        self.patch_size = getattr(backbone, "patch_size", 14)

        # Freeze backbone entirely
        self.backbone = backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Single linear classifier (1x1 conv) on top of backbone features
        self.classifier = nn.Conv2d(hidden_size, num_classes, kernel_size=1)

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward pass.

        Args:
            pixel_values: Input images [B, C, H, W]
            labels: Ground truth segmentation masks [B, H, W]

        Returns:
            Dictionary with 'logits' and optionally 'loss'
        """
        original_size = None

        if self.auto_pad:
            pixel_values, labels, original_size = pad_to_patch_size(
                pixel_values=pixel_values,
                labels=labels,
                patch_size=self.patch_size,
                ignore_value=self.loss_ignore_index,
            )

        # Extract features with frozen backbone
        with torch.no_grad():
            backbone_output = self.backbone(pixel_values)

        # Use the last feature map from the backbone
        features = backbone_output.feature_maps[-1]

        # Linear classifier
        logits = self.classifier(features)

        # Upsample logits to input resolution
        logits = F.interpolate(
            logits,
            size=pixel_values.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        # Crop back to original size if padding was applied
        if original_size is not None:
            orig_h, orig_w = original_size
            logits = logits[:, :, :orig_h, :orig_w]

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=self.loss_ignore_index,
            )
            result["loss"] = loss

        return result

    def get_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get segmentation logits without computing loss."""
        return self.forward(pixel_values)["logits"]

    def freeze_backbone(self):
        """Freeze all backbone parameters (already frozen by default)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    @property
    def backbone_module(self) -> nn.Module:
        """Access the backbone module."""
        return self.backbone
