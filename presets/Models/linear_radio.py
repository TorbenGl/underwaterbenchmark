"""
NVIDIA RADIO linear probing presets for semantic segmentation.

Frozen RADIO backbone + single 1x1 conv linear classifier.
Only the last transformer layer is used as feature source.
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones import UperNetBackboneAdapter
from models.linear_probing_model import LinearProbingSegmentationModel


class LinearRadioPreset:
    """Base preset class for linear probing with RADIO backbones."""

    name: str = "linear_radio_v3_base"
    model_name: str = "c-radio_v3-b"
    backbone_indices: List[int] = [11]
    scales: List[float] = [1.0]
    out_channels: int = 512

    @classmethod
    def create(
        cls,
        num_classes: int,
        img_size: tuple[int, int],
        loss_ignore_index: int = 255,
        use_auxiliary_head: bool = False,
        **kwargs,
    ) -> nn.Module:
        backbone = UperNetBackboneAdapter.from_radio(
            model_name=cls.model_name,
            backbone_indices=cls.backbone_indices,
            scales=cls.scales,
            out_channels=cls.out_channels,
            img_size=img_size[0],
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
            "backbone_indices": cls.backbone_indices,
            "scales": cls.scales,
            "out_channels": cls.out_channels,
        }


# =============================================================================
# LINEAR PROBING + C-RADIO V4 (Commercially permissive)
# =============================================================================

@register_model
class LinearRadioV4H(LinearRadioPreset):
    """Linear probing with C-RADIOv4-H backbone (last layer only)."""
    name = "linear_radio_v4_h"
    model_name = "c-radio_v4-h"
    backbone_indices = [31]
    scales = [1.0]
    out_channels = 768


@register_model
class LinearRadioV4SO400M(LinearRadioPreset):
    """Linear probing with C-RADIOv4-SO400M backbone (last layer only)."""
    name = "linear_radio_v4_so400m"
    model_name = "c-radio_v4-so400m"
    backbone_indices = [26]
    scales = [1.0]
    out_channels = 768


# =============================================================================
# LINEAR PROBING + C-RADIO V3 (NVIDIA Open Model License)
# =============================================================================

@register_model
class LinearRadioV3Base(LinearRadioPreset):
    """Linear probing with C-RADIOv3-B backbone (last layer only)."""
    name = "linear_radio_v3_base"
    model_name = "c-radio_v3-b"
    backbone_indices = [11]
    scales = [1.0]
    out_channels = 512


@register_model
class LinearRadioV3Large(LinearRadioPreset):
    """Linear probing with C-RADIOv3-L backbone (last layer only)."""
    name = "linear_radio_v3_large"
    model_name = "c-radio_v3-l"
    backbone_indices = [23]
    scales = [1.0]
    out_channels = 768


@register_model
class LinearRadioV3Huge(LinearRadioPreset):
    """Linear probing with C-RADIOv3-H backbone (last layer only)."""
    name = "linear_radio_v3_huge"
    model_name = "c-radio_v3-h"
    backbone_indices = [31]
    scales = [1.0]
    out_channels = 768


@register_model
class LinearRadioV3Giant(LinearRadioPreset):
    """Linear probing with C-RADIOv3-g backbone (last layer only)."""
    name = "linear_radio_v3_giant"
    model_name = "c-radio_v3-g"
    backbone_indices = [39]
    scales = [1.0]
    out_channels = 1024
