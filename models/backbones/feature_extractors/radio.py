"""
NVIDIA RADIO feature extractor.

Handles loading RADIO models via torch.hub and extracting raw features.
RADIO combines knowledge from CLIP, DINOv2, and SAM teachers.

Reference: https://github.com/NVlabs/RADIO
"""

from typing import List

import torch
import torch.nn as nn

from models.backbones.feature_extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)


# Model configurations for RADIO variants
RADIO_CONFIGS = {
    # C-RADIOv4 (commercially permissive)
    "c-radio_v4-h": {
        "version": "c-radio_v4-h",
        "hidden_size": 1280,
        "patch_size": 16,
        "num_layers": 32,
    },
    "c-radio_v4-so400m": {
        "version": "c-radio_v4-so400m",
        "hidden_size": 1152,
        "patch_size": 14,
        "num_layers": 27,
    },
    # C-RADIOv3 (commercially viable)
    "c-radio_v3-b": {
        "version": "c-radio_v3-b",
        "hidden_size": 768,
        "patch_size": 16,
        "num_layers": 12,
    },
    "c-radio_v3-l": {
        "version": "c-radio_v3-l",
        "hidden_size": 1024,
        "patch_size": 16,
        "num_layers": 24,
    },
    "c-radio_v3-h": {
        "version": "c-radio_v3-h",
        "hidden_size": 1280,
        "patch_size": 16,
        "num_layers": 32,
    },
    "c-radio_v3-g": {
        "version": "c-radio_v3-g",
        "hidden_size": 1536,
        "patch_size": 14,
        "num_layers": 40,
    },
    # Legacy RADIO models (non-commercial)
    "radio_v2.5-l": {
        "version": "radio_v2.5-l",
        "hidden_size": 1024,
        "patch_size": 16,
        "num_layers": 24,
    },
    "radio_v2.5-h": {
        "version": "radio_v2.5-h",
        "hidden_size": 1280,
        "patch_size": 16,
        "num_layers": 32,
    },
}


class RadioFeatureExtractor(FeatureExtractor):
    """
    Feature extractor for NVIDIA RADIO models.

    RADIO models are loaded via torch.hub and return:
        - summary: Global feature vector [B, D]
        - spatial_features: Spatial features [B, C, H, W] when using NCHW format

    Note: RADIO doesn't expose intermediate transformer layers, so the same
    final features are returned for all requested indices.

    Example:
        >>> extractor = RadioFeatureExtractor.from_pretrained("c-radio_v4-h")
        >>> features = extractor.extract_features(pixel_values, indices=[0, 1, 2, 3])
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
    ):
        """
        Initialize the RADIO feature extractor.

        Args:
            model: RADIO model loaded via torch.hub
            hidden_size: Hidden dimension of the model
            patch_size: Patch size
            num_layers: Number of transformer layers (for reference)
        """
        super().__init__()
        self._model = model
        self._hidden_size = hidden_size
        self._patch_size = patch_size
        self._num_layers = num_layers

        # Update patch_size from model if available
        if hasattr(model, 'patch_size'):
            self._patch_size = model.patch_size

    @property
    def config(self) -> FeatureExtractorConfig:
        return FeatureExtractorConfig(
            hidden_size=self._hidden_size,
            patch_size=self._patch_size,
            num_layers=self._num_layers,
            num_special_tokens=0,  # RADIO returns spatial features without CLS
            supports_intermediate=False,  # RADIO doesn't expose intermediate layers
        )

    def extract_features(
        self,
        pixel_values: torch.Tensor,
        indices: List[int],
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> ExtractedFeatures:
        """
        Extract features from the RADIO model.

        RADIO returns (summary, spatial_features) where spatial_features
        are in NCHW format. We convert to sequence format [B, N, C] to match
        other extractors.

        Args:
            pixel_values: Input images [B, C, H, W]
            indices: Layer indices (ignored - same features for all)
            output_hidden_states: Not used (RADIO doesn't expose intermediate states)
            output_attentions: Not used (RADIO doesn't expose attention weights)

        Returns:
            ExtractedFeatures with same features duplicated for each index
        """
        with torch.set_grad_enabled(self.training):
            summary, spatial_features = self._model(pixel_values, feature_fmt='NCHW')

        # Convert from NCHW [B, C, H, W] to sequence [B, N, C]
        B, C, H, W = spatial_features.shape
        spatial_seq = spatial_features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # RADIO doesn't have CLS token, so we don't add one
        # The UperNetBackboneAdapter needs to handle this (num_special_tokens=0)

        # Duplicate features for each requested index
        features = [spatial_seq for _ in indices]

        return ExtractedFeatures(
            features=features,
            hidden_states=None,
            attentions=None,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        **kwargs,
    ) -> "RadioFeatureExtractor":
        """
        Create a RadioFeatureExtractor from a pretrained model via torch.hub.

        Args:
            model_name: Model name (e.g., "c-radio_v4-h", "c-radio_v3-l")
            **kwargs: Additional arguments

        Returns:
            RadioFeatureExtractor instance
        """
        if model_name not in RADIO_CONFIGS:
            available = ", ".join(RADIO_CONFIGS.keys())
            raise ValueError(f"Unknown RADIO model: {model_name}. Available: {available}")

        config = RADIO_CONFIGS[model_name]

        # Load model via torch.hub
        model = torch.hub.load(
            'NVlabs/RADIO',
            'radio_model',
            version=config["version"],
            progress=True,
        )
        model.eval()

        return cls(
            model=model,
            hidden_size=config["hidden_size"],
            patch_size=config["patch_size"],
            num_layers=config["num_layers"],
        )
