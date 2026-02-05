"""
UperNet Semantic Segmentation Model.

This module provides the UperNet segmentation model that combines a ViT backbone
with the UperNet decoder from HuggingFace transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import UperNetForSemanticSegmentation, UperNetConfig
from typing import Optional, List, Dict, Any, Tuple


def pad_to_patch_size(
    pixel_values: torch.Tensor,
    labels: Optional[torch.Tensor],
    patch_size: int,
    ignore_value: int = 255,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
    """
    Pad image and mask to be divisible by patch_size.

    Args:
        pixel_values: Image tensor of shape (B, C, H, W)
        labels: Mask tensor of shape (B, H, W) or None
        patch_size: The patch size to pad to (e.g., 14 for ViT)
        ignore_value: Value to use for mask padding (default: 255)

    Returns:
        Tuple of (padded_image, padded_mask, original_size)
    """
    _, _, h, w = pixel_values.shape
    original_size = (h, w)

    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    if pad_h == 0 and pad_w == 0:
        return pixel_values, labels, original_size

    # Pad image with reflection (preserves visual continuity)
    # F.pad format: (left, right, top, bottom)
    pixel_values_padded = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="reflect")

    # Pad mask with ignore_value (so padded regions are ignored in loss)
    labels_padded = None
    if labels is not None:
        labels_padded = F.pad(
            labels, (0, pad_w, 0, pad_h), mode="constant", value=ignore_value
        )

    return pixel_values_padded, labels_padded, original_size


class UperNetSegmentationModel(nn.Module):
    """
    UperNet model for semantic segmentation with a configurable backbone.

    This model combines:
    - A ViT-based backbone (UperVitBackbone) for feature extraction
    - UperNet decoder for semantic segmentation

    Args:
        backbone: The backbone module (e.g., UperVitBackbone)
        num_classes: Number of segmentation classes
        hidden_size: Hidden size for UperNet decoder
        hidden_sizes: List of hidden sizes for each feature level
        use_auxiliary_head: Whether to use auxiliary segmentation head
        loss_ignore_index: Index to ignore in loss computation
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

        # Get patch size from backbone for auto-padding
        self.patch_size = getattr(backbone, "patch_size", 14)

        # Build UperNet config
        config = UperNetConfig(
            backbone_config=None,
            use_auxiliary_head=use_auxiliary_head,
            hidden_size=hidden_size,
            loss_ignore_index=loss_ignore_index,
            num_labels=num_classes,
        )

        # Set per-stage hidden sizes
        if hidden_sizes is not None:
            config.backbone_config.hidden_sizes = hidden_sizes

        # Initialize UperNet and replace backbone
        self.model = UperNetForSemanticSegmentation(config)
        self.model.backbone = backbone

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

        # Auto-pad to patch size if enabled
        if self.auto_pad:
            pixel_values, labels, original_size = pad_to_patch_size(
                pixel_values=pixel_values,
                labels=labels,
                patch_size=self.patch_size,
                ignore_value=self.loss_ignore_index,
            )

        outputs = self.model(pixel_values=pixel_values, labels=labels)

        logits = outputs.logits

        # Crop logits back to original size if padding was applied
        if original_size is not None:
            orig_h, orig_w = original_size
            logits = logits[:, :, :orig_h, :orig_w]

        result = {"logits": logits}

        if outputs.loss is not None:
            result["loss"] = outputs.loss

        return result

    def get_logits(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get segmentation logits without computing loss."""
        original_size = None

        # Auto-pad to patch size if enabled
        if self.auto_pad:
            pixel_values, _, original_size = pad_to_patch_size(
                pixel_values=pixel_values,
                labels=None,
                patch_size=self.patch_size,
                ignore_value=self.loss_ignore_index,
            )

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits

        # Crop logits back to original size if padding was applied
        if original_size is not None:
            orig_h, orig_w = original_size
            logits = logits[:, :, :orig_h, :orig_w]

        return logits

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True

    @property
    def backbone(self) -> nn.Module:
        """Access the backbone module."""
        return self.model.backbone
