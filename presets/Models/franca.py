"""
Franca model presets for semantic segmentation.

Franca is a vision foundation model from Valeo AI with:
- Nested Matryoshka Clustering for hierarchical features
- RASA (Relative Absolute Spatial Attention) for position-debiased representations
- CyclicMask masking strategy

Reference: https://github.com/valeoai/Franca

Available models:
- franca_vitb14: ViT-B/14, 86M params, 768 dim
- franca_vitl14: ViT-L/14, 300M params, 1024 dim
- franca_vitg14: ViT-g/14, 1.1B params, 1536 dim

Each has two weight variants:
- IN21K: Trained on ImageNet-21K
- LAION: Trained on LAION-600M
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones import UperNetBackboneAdapter
from models.UperNetModel import UperNetSegmentationModel


class FrancaModelPreset:
    """
    Base preset class for Franca models.

    Uses UperNetBackboneAdapter.from_franca() to create the backbone
    with a FrancaFeatureExtractor.
    """

    name: str = "upernet_franca_base"
    model_name: str = "franca_vitb14"
    weights: str = "IN21K"
    backbone_indices: List[int] = [2, 5, 8, 11]
    scales: List[float] = [4.0, 2.0, 1.0, 0.5]
    out_channels: int = 512
    use_rasa_head: bool = True

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
            The segmentation model
        """
        # Create backbone using UperNetBackboneAdapter with Franca extractor
        backbone = UperNetBackboneAdapter.from_franca(
            model_name=cls.model_name,
            backbone_indices=cls.backbone_indices,
            scales=cls.scales,
            out_channels=cls.out_channels,
            weights=cls.weights,
            use_rasa_head=cls.use_rasa_head,
            img_size=img_size[0],
        )

        # Hidden sizes for UperNet
        hidden_sizes = [cls.out_channels] * len(cls.scales)

        # Create segmentation model
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
    def get_info(cls):
        """Get preset information."""
        return {
            "name": cls.name,
            "model": cls.model_name,
            "weights": cls.weights,
            "backbone_indices": cls.backbone_indices,
            "scales": cls.scales,
            "out_channels": cls.out_channels,
            "use_rasa_head": cls.use_rasa_head,
        }


# =============================================================================
# UPERNET + FRANCA PRESETS (ImageNet-21K weights)
# =============================================================================

@register_model
class UperNetFrancaBase(FrancaModelPreset):
    """UperNet with Franca ViT-B/14 backbone (ImageNet-21K weights).

    Architecture: 86M params, 12 layers, embed_dim=768.
    Performance: 77.4% k-NN, 82.0% linear probe on ImageNet-1K.
    """
    name = "upernet_franca_base"
    model_name = "franca_vitb14"
    weights = "IN21K"
    backbone_indices = [2, 5, 8, 11]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 512
    use_rasa_head = True


@register_model
class UperNetFrancaLarge(FrancaModelPreset):
    """UperNet with Franca ViT-L/14 backbone (ImageNet-21K weights).

    Architecture: 300M params, 24 layers, embed_dim=1024.
    Performance: 82.2% k-NN, 84.5% linear probe on ImageNet-1K.
    """
    name = "upernet_franca_large"
    model_name = "franca_vitl14"
    weights = "IN21K"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768
    use_rasa_head = True


@register_model
class UperNetFrancaGiant(FrancaModelPreset):
    """UperNet with Franca ViT-g/14 backbone (ImageNet-21K weights).

    Architecture: 1.1B params, 40 layers, embed_dim=1536.
    Performance: 83.0% k-NN, 85.9% linear probe on ImageNet-1K.
    """
    name = "upernet_franca_giant"
    model_name = "franca_vitg14"
    weights = "IN21K"
    backbone_indices = [9, 19, 29, 39]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 1024
    use_rasa_head = True


# =============================================================================
# UPERNET + FRANCA PRESETS (LAION-600M weights)
# =============================================================================

@register_model
class UperNetFrancaLargeLAION(FrancaModelPreset):
    """UperNet with Franca ViT-L/14 backbone (LAION-600M weights).

    Architecture: 300M params, 24 layers, embed_dim=1024.
    Performance: 81.9% k-NN, 84.4% linear probe on ImageNet-1K.
    """
    name = "upernet_franca_large_laion"
    model_name = "franca_vitl14"
    weights = "LAION"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768
    use_rasa_head = True


@register_model
class UperNetFrancaGiantLAION(FrancaModelPreset):
    """UperNet with Franca ViT-g/14 backbone (LAION-600M weights).

    Architecture: 1.1B params, 40 layers, embed_dim=1536.
    Performance: 81.2% k-NN, 85.0% linear probe on ImageNet-1K.
    """
    name = "upernet_franca_giant_laion"
    model_name = "franca_vitg14"
    weights = "LAION"
    backbone_indices = [9, 19, 29, 39]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 1024
    use_rasa_head = True


# =============================================================================
# UPERNET + FRANCA PRESETS (No RASA - standard features)
# =============================================================================

@register_model
class UperNetFrancaLargeNoRASA(FrancaModelPreset):
    """UperNet with Franca ViT-L/14 without RASA head.

    Uses standard patch token features instead of position-debiased RASA features.
    Useful for comparison or when position information is important.
    """
    name = "upernet_franca_large_no_rasa"
    model_name = "franca_vitl14"
    weights = "IN21K"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768
    use_rasa_head = False
