"""
Backbone adapters for semantic segmentation with UperNet.

Architecture:
    FeatureExtractor (model-specific)     UperNetBackboneAdapter (decoder-specific)
    ├── DinoFeatureExtractor         ───►  Composes any FeatureExtractor
    ├── FrancaFeatureExtractor             - Reshapes [B, N, C] → [B, C, H, W]
    └── RadioFeatureExtractor              - Lateral convs for channel projection
                                           - Scales to different resolutions
                                           - Smoothing convolutions

Usage:
    # Option 1: Using factory methods (recommended)
    from models.backbones import UperNetBackboneAdapter

    backbone = UperNetBackboneAdapter.from_dino(
        "facebook/dinov2-base",
        backbone_indices=[2, 5, 8, 11],
        scales=[4.0, 2.0, 1.0, 0.5],
        out_channels=512,
    )

    # Option 2: Compose manually
    from models.backbones import UperNetBackboneAdapter
    from models.backbones.feature_extractors import DinoFeatureExtractor

    extractor = DinoFeatureExtractor.from_pretrained("facebook/dinov2-base")
    backbone = UperNetBackboneAdapter(
        feature_extractor=extractor,
        backbone_indices=[2, 5, 8, 11],
        ...
    )
"""

# Main adapters
from models.backbones.upernet_adapter import UperNetBackboneAdapter
from models.backbones.resnet_adapter import ResNetBackboneAdapter
from models.backbones.classification_adapter import ClassificationBackboneAdapter

# Feature extractors
from models.backbones.feature_extractors import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
    DinoFeatureExtractor,
    FrancaFeatureExtractor,
    RadioFeatureExtractor,
    FRANCA_CONFIGS,
    RADIO_CONFIGS,
)

__all__ = [
    # Main adapters
    "UperNetBackboneAdapter",
    "ResNetBackboneAdapter",
    "ClassificationBackboneAdapter",
    # Feature extractors
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "ExtractedFeatures",
    "DinoFeatureExtractor",
    "FrancaFeatureExtractor",
    "RadioFeatureExtractor",
    "FRANCA_CONFIGS",
    "RADIO_CONFIGS",
]
