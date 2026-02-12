"""
Classification Backbone Adapter.

This adapter composes a FeatureExtractor for image classification tasks.
It extracts features from a single transformer layer and pools them to
produce a single feature vector per image.

Two pooling modes are supported:
- "cls": Use the CLS token directly (linear probing)
- "gap": Global average pooling over patch tokens
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.feature_extractors.base import FeatureExtractor
from models.upernet_model import pad_to_patch_size


class ClassificationBackboneAdapter(nn.Module):
    """
    Backbone adapter that composes a FeatureExtractor for classification.

    Unlike UperNetBackboneAdapter which produces multi-scale spatial feature maps,
    this adapter extracts features from a single layer and pools them to a
    single vector [B, hidden_size] suitable for classification.

    Args:
        feature_extractor: FeatureExtractor instance
        backbone_index: Transformer layer index to extract from (typically the last)
        pooling_mode: "cls" for CLS token, "gap" for global average pooling
        img_size: Input image size (for patch size calculation)

    Example:
        >>> adapter = ClassificationBackboneAdapter.from_dino(
        ...     "facebook/dinov3-vitb16-pretrain-lvd1689m",
        ...     backbone_index=11,
        ...     pooling_mode="cls",
        ... )
        >>> features = adapter(pixel_values)  # [B, hidden_size]
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        backbone_index: int,
        pooling_mode: str = "cls",
        img_size: int = 224,
    ):
        super().__init__()

        assert pooling_mode in ("cls", "gap"), \
            f"pooling_mode must be 'cls' or 'gap', got '{pooling_mode}'"

        self.feature_extractor = feature_extractor
        self.backbone_index = backbone_index
        self.pooling_mode = pooling_mode

        config = feature_extractor.config
        self.hidden_size = config.hidden_size
        self.patch_size = config.patch_size
        self.num_special_tokens = config.num_special_tokens

        if pooling_mode == "cls" and self.num_special_tokens == 0:
            # RADIO with return_prefix_tokens=False has 0 special tokens,
            # but we'll request prefix tokens during forward pass
            pass

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract and pool features from the backbone.

        Args:
            pixel_values: Input images [B, C, H, W]

        Returns:
            Feature vector [B, hidden_size]
        """
        # Build kwargs for extract_features
        extract_kwargs = {
            "pixel_values": pixel_values,
            "indices": [self.backbone_index],
        }

        # For CLS mode, request prefix tokens from RADIO
        if self.pooling_mode == "cls":
            extract_kwargs["return_prefix_tokens"] = True

        extracted = self.feature_extractor.extract_features(**extract_kwargs)
        features = extracted.features[0]  # [B, N, C]

        if self.pooling_mode == "cls":
            # CLS token is always at position 0
            return features[:, 0, :]
        else:
            # GAP: strip special tokens, mean-pool patch tokens
            patch_tokens = features[:, self.num_special_tokens:, :]
            return patch_tokens.mean(dim=1)

    def freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    @classmethod
    def from_dino(
        cls,
        model_name: str,
        backbone_index: int,
        pooling_mode: str = "cls",
        num_register_tokens: int = 0,
        img_size: int = 224,
        **kwargs,
    ) -> "ClassificationBackboneAdapter":
        """
        Create adapter with a DINO feature extractor.

        Args:
            model_name: HuggingFace model name (e.g., "facebook/dinov3-vitb16-pretrain-lvd1689m")
            backbone_index: Layer index to extract features from
            pooling_mode: "cls" or "gap"
            num_register_tokens: Number of register tokens
            img_size: Input image size
        """
        from models.backbones.feature_extractors import DinoFeatureExtractor

        extractor = DinoFeatureExtractor.from_pretrained(
            model_name,
            num_register_tokens=num_register_tokens,
            **kwargs,
        )

        return cls(
            feature_extractor=extractor,
            backbone_index=backbone_index,
            pooling_mode=pooling_mode,
            img_size=img_size,
        )

    @classmethod
    def from_franca(
        cls,
        model_name: str,
        backbone_index: int,
        pooling_mode: str = "cls",
        weights: str = "IN21K",
        use_rasa_head: bool = True,
        **kwargs,
    ) -> "ClassificationBackboneAdapter":
        """
        Create adapter with a Franca feature extractor.

        Args:
            model_name: Model name (e.g., "franca_vitl14")
            backbone_index: Layer index to extract features from
            pooling_mode: "cls" or "gap"
            weights: Weight variant ("IN21K" or "LAION")
            use_rasa_head: Whether to use RASA head
        """
        from models.backbones.feature_extractors import FrancaFeatureExtractor

        extractor = FrancaFeatureExtractor.from_pretrained(
            model_name,
            weights=weights,
            use_rasa_head=use_rasa_head,
            **kwargs,
        )

        return cls(
            feature_extractor=extractor,
            backbone_index=backbone_index,
            pooling_mode=pooling_mode,
            img_size=518,
        )

    @classmethod
    def from_radio(
        cls,
        model_name: str,
        backbone_index: Optional[int] = None,
        pooling_mode: str = "cls",
        img_size: int = 224,
        **kwargs,
    ) -> "ClassificationBackboneAdapter":
        """
        Create adapter with a RADIO feature extractor.

        Args:
            model_name: Model name (e.g., "c-radio_v4-h")
            backbone_index: Layer index to extract features from.
                If None, uses the last layer.
            pooling_mode: "cls" or "gap"
            img_size: Input image size
        """
        from models.backbones.feature_extractors import RadioFeatureExtractor
        from models.backbones.feature_extractors.radio import RADIO_CONFIGS

        extractor = RadioFeatureExtractor.from_pretrained(
            model_name,
            **kwargs,
        )

        if backbone_index is None:
            backbone_index = RADIO_CONFIGS[model_name]["num_layers"] - 1

        return cls(
            feature_extractor=extractor,
            backbone_index=backbone_index,
            pooling_mode=pooling_mode,
            img_size=img_size,
        )
