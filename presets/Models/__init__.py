"""
Model presets for semantic segmentation.

Usage:
    from presets.Models import get_model, AVAILABLE_MODELS

    # Get a model by name
    model = get_model(
        name="upernet_vit_base",
        num_classes=25,
        img_size=(512, 512),
    )
"""

from typing import Dict, Type, Optional, List
import torch.nn as nn

from models.backbones import UperNetBackboneAdapter
from models.upernet_model import UperNetSegmentationModel


# =============================================================================
# MODEL REGISTRY
# =============================================================================

_MODEL_REGISTRY: Dict[str, Type["ModelPreset"]] = {}


def register_model(cls: Type["ModelPreset"]) -> Type["ModelPreset"]:
    """Decorator to register a model preset."""
    _MODEL_REGISTRY[cls.name] = cls
    return cls


def get_model(
    name: str,
    num_classes: int,
    img_size: tuple[int, int],
    **kwargs,
) -> nn.Module:
    """
    Get a model by preset name.

    Args:
        name: Model preset name
        num_classes: Number of output classes
        img_size: Input image size (H, W)
        **kwargs: Additional arguments passed to the preset

    Returns:
        The model
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: '{name}'. Available models: {available}")

    preset_cls = _MODEL_REGISTRY[name]
    return preset_cls.create(num_classes=num_classes, img_size=img_size, **kwargs)


def list_models() -> List[str]:
    """List all available model names."""
    return list(_MODEL_REGISTRY.keys())


# =============================================================================
# BASE MODEL PRESET
# =============================================================================

class ModelPreset:
    """
    Base class for model presets.

    Subclasses should define:
        - name: Model identifier
        - backbone_name: HuggingFace model name
        - backbone_indices: Layer indices for feature extraction
        - scales: Spatial scales for each feature level
        - out_channels: Output channels for decoder
        - num_register_tokens: Number of register tokens (0 for standard ViT)
    """

    name: str = "base"
    backbone_name: str = "google/vit-base-patch16-224"
    backbone_indices: List[int] = [2, 5, 8, 11]
    scales: List[float] = [4.0, 2.0, 1.0, 0.5]
    out_channels: int = 512
    num_register_tokens: int = 0

    @classmethod
    def create(
        cls,
        num_classes: int,
        img_size: tuple[int, int],
        loss_ignore_index: int = 255,
        use_auxiliary_head: bool = False,
        **kwargs,
    ) -> nn.Module:
        """
        Create the model.

        Args:
            num_classes: Number of output classes
            img_size: Input image size (H, W)
            loss_ignore_index: Index to ignore in loss
            use_auxiliary_head: Whether to use auxiliary head
            **kwargs: Additional arguments

        Returns:
            The model
        """
        # Create backbone using UperNetBackboneAdapter with DINO extractor
        backbone = UperNetBackboneAdapter.from_dino(
            model_name=cls.backbone_name,
            backbone_indices=cls.backbone_indices,
            scales=cls.scales,
            out_channels=cls.out_channels,
            num_register_tokens=cls.num_register_tokens,
            img_size=img_size[0],
        )

        # Hidden sizes for UperNet
        hidden_sizes = [cls.out_channels] * len(cls.backbone_indices)

        # Create model
        model = UperNetSegmentationModel(
            backbone=backbone,
            num_classes=num_classes,
            hidden_size=cls.out_channels,
            hidden_sizes=hidden_sizes,
            use_auxiliary_head=use_auxiliary_head,
            loss_ignore_index=loss_ignore_index,
        )

        return model

    @classmethod
    def get_info(cls) -> Dict[str, any]:
        """Get preset information."""
        return {
            "name": cls.name,
            "backbone": cls.backbone_name,
            "backbone_indices": cls.backbone_indices,
            "scales": cls.scales,
            "out_channels": cls.out_channels,
            "num_register_tokens": cls.num_register_tokens,
        }


# =============================================================================
# IMPORT PRESET MODULES (registers models via decorators)
# =============================================================================

from presets.Models import dino  # noqa: E402, F401
from presets.Models import franca  # noqa: E402, F401
from presets.Models import radio  # noqa: E402, F401


# For convenience
AVAILABLE_MODELS = list_models()
