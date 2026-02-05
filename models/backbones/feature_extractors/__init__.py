"""
Feature extractors for different vision foundation models.

Feature extractors handle model-specific loading and raw feature extraction.
They are composed with UperNetBackboneAdapter which handles the common
reshaping, projection, and scaling logic.

Available extractors:
- DinoFeatureExtractor: DINOv2/v3 models (HuggingFace)
- FrancaFeatureExtractor: Franca models (torch.hub)
- RadioFeatureExtractor: NVIDIA RADIO models (torch.hub)

Usage:
    from models.backbones.feature_extractors import DinoFeatureExtractor

    extractor = DinoFeatureExtractor.from_pretrained("facebook/dinov2-base")
    features = extractor.extract_features(pixel_values, indices=[2, 5, 8, 11])
"""

from models.backbones.feature_extractors.base import (
    FeatureExtractor,
    FeatureExtractorConfig,
    ExtractedFeatures,
)
from models.backbones.feature_extractors.dino import DinoFeatureExtractor
from models.backbones.feature_extractors.franca import FrancaFeatureExtractor, FRANCA_CONFIGS
from models.backbones.feature_extractors.radio import RadioFeatureExtractor, RADIO_CONFIGS

__all__ = [
    # Base classes
    "FeatureExtractor",
    "FeatureExtractorConfig",
    "ExtractedFeatures",
    # Extractors
    "DinoFeatureExtractor",
    "FrancaFeatureExtractor",
    "RadioFeatureExtractor",
    # Configs
    "FRANCA_CONFIGS",
    "RADIO_CONFIGS",
]
