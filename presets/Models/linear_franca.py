"""
Franca linear probing presets for semantic segmentation.

Frozen Franca backbone + single 1x1 conv linear classifier.
Only the last transformer layer is used as feature source.
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones import UperNetBackboneAdapter
from models.linear_probing_model import LinearProbingSegmentationModel


class LinearFrancaPreset:
    """Base preset class for linear probing with Franca backbones."""

    name: str = "linear_franca_base"
    model_name: str = "franca_vitb14"
    weights: str = "IN21K"
    backbone_indices: List[int] = [11]
    scales: List[float] = [1.0]
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
        backbone = UperNetBackboneAdapter.from_franca(
            model_name=cls.model_name,
            backbone_indices=cls.backbone_indices,
            scales=cls.scales,
            out_channels=cls.out_channels,
            weights=cls.weights,
            use_rasa_head=cls.use_rasa_head,
        )

        hidden_sizes = [cls.out_channels] * len(cls.backbone_indices)

        model = LinearProbingSegmentationModel(
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
        return {
            "name": cls.name,
            "backbone": cls.model_name,
            "weights": cls.weights,
            "backbone_indices": cls.backbone_indices,
            "scales": cls.scales,
            "out_channels": cls.out_channels,
            "use_rasa_head": cls.use_rasa_head,
        }


# =============================================================================
# LINEAR PROBING + FRANCA (ImageNet-21K, with RASA)
# =============================================================================

@register_model
class LinearFrancaBase(LinearFrancaPreset):
    """Linear probing with Franca ViT-B/14 backbone (IN21K, RASA, last layer only)."""
    name = "linear_franca_base"
    model_name = "franca_vitb14"
    weights = "IN21K"
    backbone_indices = [11]
    scales = [1.0]
    out_channels = 512
    use_rasa_head = True


# =============================================================================
# LINEAR PROBING + FRANCA (LAION-600M, with RASA)
# =============================================================================

@register_model
class LinearFrancaLargeLAION(LinearFrancaPreset):
    """Linear probing with Franca ViT-L/14 backbone (LAION, RASA, last layer only)."""
    name = "linear_franca_large_laion"
    model_name = "franca_vitl14"
    weights = "LAION"
    backbone_indices = [23]
    scales = [1.0]
    out_channels = 768
    use_rasa_head = True


@register_model
class LinearFrancaGiantLAION(LinearFrancaPreset):
    """Linear probing with Franca ViT-g/14 backbone (LAION, RASA, last layer only)."""
    name = "linear_franca_giant_laion"
    model_name = "franca_vitg14"
    weights = "LAION"
    backbone_indices = [39]
    scales = [1.0]
    out_channels = 1024
    use_rasa_head = True


# =============================================================================
# LINEAR PROBING + FRANCA (No RASA)
# =============================================================================

@register_model
class LinearFrancaBaseNoRASA(LinearFrancaPreset):
    """Linear probing with Franca ViT-B/14 without RASA (last layer only)."""
    name = "linear_franca_base_no_rasa"
    model_name = "franca_vitb14"
    weights = "IN21K"
    backbone_indices = [11]
    scales = [1.0]
    out_channels = 512
    use_rasa_head = False


@register_model
class LinearFrancaLargeNoRASA(LinearFrancaPreset):
    """Linear probing with Franca ViT-L/14 (LAION) without RASA (last layer only)."""
    name = "linear_franca_large_no_rasa"
    model_name = "franca_vitl14"
    weights = "LAION"
    backbone_indices = [23]
    scales = [1.0]
    out_channels = 768
    use_rasa_head = False


@register_model
class LinearFrancaGiantNoRASA(LinearFrancaPreset):
    """Linear probing with Franca ViT-g/14 (LAION) without RASA (last layer only)."""
    name = "linear_franca_giant_no_rasa"
    model_name = "franca_vitg14"
    weights = "LAION"
    backbone_indices = [39]
    scales = [1.0]
    out_channels = 1024
    use_rasa_head = False
