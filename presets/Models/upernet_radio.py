"""
NVIDIA RADIO model presets for semantic segmentation.

RADIO (Robust And DIverse grOunding) is NVIDIA's vision foundation model that:
- Combines knowledge from CLIP, DINOv2, and SAM teachers
- Supports arbitrary input resolutions
- Provides both summary and spatial features

Reference: https://github.com/NVlabs/RADIO

Model families:
- C-RADIOv4: Latest commercially permissive models
- C-RADIOv3: Commercial models (NVIDIA Open Model License)
- RADIO v2.5: Legacy models (non-commercial)

Sizes:
- B (Base): ~86M params
- L (Large): ~300M params
- H (Huge): ~632M params
- g (Giant): ~1.1B params
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones import UperNetBackboneAdapter
from models.upernet_model import UperNetSegmentationModel


class RadioModelPreset:
    """
    Base preset class for RADIO models.

    Uses UperNetBackboneAdapter.from_radio() to create the backbone
    with a RadioFeatureExtractor. Supports intermediate layer extraction
    via backbone_indices (evenly spaced across transformer depth).
    """

    name: str = "upernet_radio_base"
    model_name: str = "c-radio_v3-b"
    backbone_indices: List[int] = [2, 5, 8, 11]
    scales: List[float] = [4.0, 2.0, 1.0, 0.5]
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
        # Create backbone using UperNetBackboneAdapter with RADIO extractor
        backbone = UperNetBackboneAdapter.from_radio(
            model_name=cls.model_name,
            backbone_indices=cls.backbone_indices,
            scales=cls.scales,
            out_channels=cls.out_channels,
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
            "backbone": cls.model_name,
            "backbone_indices": cls.backbone_indices,
            "scales": cls.scales,
            "out_channels": cls.out_channels,
        }


# =============================================================================
# UPERNET + C-RADIO V4 PRESETS (Commercially permissive)
# =============================================================================

@register_model
class UperNetRadioV4H(RadioModelPreset):
    """UperNet with C-RADIOv4-H backbone.

    Latest NVIDIA foundation model, commercially permissive license.
    Architecture: 632M params, patch_size=16, embed_dim=1280.
    """
    name = "upernet_radio_v4_h"
    model_name = "c-radio_v4-h"
    backbone_indices = [7, 15, 23, 31]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768


@register_model
class UperNetRadioV4HLastLayer(UperNetRadioV4H):
    """UperNet with C-RADIOv4-H backbone -- last-layer-only features."""
    name = "upernet_radio_v4_h_lastlayer"
    backbone_indices = [31, 31, 31, 31]


@register_model
class UperNetRadioV4SO400M(RadioModelPreset):
    """UperNet with C-RADIOv4-SO400M backbone.

    Efficient NVIDIA foundation model, commercially permissive license.
    Architecture: ~400M params, patch_size=14, embed_dim=1152.
    """
    name = "upernet_radio_v4_so400m"
    model_name = "c-radio_v4-so400m"
    backbone_indices = [6, 13, 19, 26]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768


@register_model
class UperNetRadioV4SO400MLastLayer(UperNetRadioV4SO400M):
    """UperNet with C-RADIOv4-SO400M backbone -- last-layer-only features."""
    name = "upernet_radio_v4_so400m_lastlayer"
    backbone_indices = [26, 26, 26, 26]


# =============================================================================
# UPERNET + C-RADIO V3 PRESETS (Commercial - NVIDIA Open Model License)
# =============================================================================

@register_model
class UperNetRadioV3Base(RadioModelPreset):
    """UperNet with C-RADIOv3-B backbone.

    Commercial NVIDIA foundation model (Base size).
    Architecture: ~86M params, patch_size=16, embed_dim=768.
    """
    name = "upernet_radio_v3_base"
    model_name = "c-radio_v3-b"
    backbone_indices = [2, 5, 8, 11]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 512


@register_model
class UperNetRadioV3BaseLastLayer(UperNetRadioV3Base):
    """UperNet with C-RADIOv3-B backbone -- last-layer-only features."""
    name = "upernet_radio_v3_base_lastlayer"
    backbone_indices = [11, 11, 11, 11]


@register_model
class UperNetRadioV3Large(RadioModelPreset):
    """UperNet with C-RADIOv3-L backbone.

    Commercial NVIDIA foundation model (Large size).
    Architecture: ~300M params, patch_size=16, embed_dim=1024.
    """
    name = "upernet_radio_v3_large"
    model_name = "c-radio_v3-l"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768


@register_model
class UperNetRadioV3LargeLastLayer(UperNetRadioV3Large):
    """UperNet with C-RADIOv3-L backbone -- last-layer-only features."""
    name = "upernet_radio_v3_large_lastlayer"
    backbone_indices = [23, 23, 23, 23]


@register_model
class UperNetRadioV3Huge(RadioModelPreset):
    """UperNet with C-RADIOv3-H backbone.

    Commercial NVIDIA foundation model (Huge size).
    Architecture: ~632M params, patch_size=16, embed_dim=1280.
    """
    name = "upernet_radio_v3_huge"
    model_name = "c-radio_v3-h"
    backbone_indices = [7, 15, 23, 31]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768


@register_model
class UperNetRadioV3HugeLastLayer(UperNetRadioV3Huge):
    """UperNet with C-RADIOv3-H backbone -- last-layer-only features."""
    name = "upernet_radio_v3_huge_lastlayer"
    backbone_indices = [31, 31, 31, 31]


@register_model
class UperNetRadioV3Giant(RadioModelPreset):
    """UperNet with C-RADIOv3-g backbone.

    Commercial NVIDIA foundation model (Giant size).
    Architecture: ~1.1B params, patch_size=14, embed_dim=1536.
    """
    name = "upernet_radio_v3_giant"
    model_name = "c-radio_v3-g"
    backbone_indices = [9, 19, 29, 39]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 1024


@register_model
class UperNetRadioV3GiantLastLayer(UperNetRadioV3Giant):
    """UperNet with C-RADIOv3-g backbone -- last-layer-only features."""
    name = "upernet_radio_v3_giant_lastlayer"
    backbone_indices = [39, 39, 39, 39]
