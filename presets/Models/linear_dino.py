"""
DINOv3 linear probing presets for semantic segmentation.

Frozen DINOv3 backbone + single 1x1 conv linear classifier.
Only the last transformer layer is used as feature source.
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones import UperNetBackboneAdapter
from models.linear_probing_model import LinearProbingSegmentationModel


class LinearDinoPreset:
    """Base preset class for linear probing with DINO backbones."""

    name: str = "linear_dinov3_base"
    backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    backbone_indices: List[int] = [11]
    scales: List[float] = [1.0]
    out_channels: int = 512
    num_register_tokens: int = 4

    @classmethod
    def create(
        cls,
        num_classes: int,
        img_size: tuple[int, int],
        loss_ignore_index: int = 255,
        use_auxiliary_head: bool = False,
        **kwargs,
    ) -> nn.Module:
        backbone = UperNetBackboneAdapter.from_dino(
            model_name=cls.backbone_name,
            backbone_indices=cls.backbone_indices,
            scales=cls.scales,
            out_channels=cls.out_channels,
            num_register_tokens=cls.num_register_tokens,
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
            "backbone": cls.backbone_name,
            "backbone_indices": cls.backbone_indices,
            "scales": cls.scales,
            "out_channels": cls.out_channels,
            "num_register_tokens": cls.num_register_tokens,
        }


@register_model
class LinearDinoV3Small(LinearDinoPreset):
    """Linear probing with DINOv3-Small backbone (last layer only)."""
    name = "linear_dinov3_small"
    backbone_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    backbone_indices = [11]
    scales = [1.0]
    out_channels = 384
    num_register_tokens = 4


@register_model
class LinearDinoV3Base(LinearDinoPreset):
    """Linear probing with DINOv3-Base backbone (last layer only)."""
    name = "linear_dinov3_base"
    backbone_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    backbone_indices = [11]
    scales = [1.0]
    out_channels = 512
    num_register_tokens = 4


@register_model
class LinearDinoV3Large(LinearDinoPreset):
    """Linear probing with DINOv3-Large backbone (last layer only)."""
    name = "linear_dinov3_large"
    backbone_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    backbone_indices = [23]
    scales = [1.0]
    out_channels = 768
    num_register_tokens = 4


@register_model
class LinearDinoV3_7B(LinearDinoPreset):
    """Linear probing with DINOv3-7B backbone (last layer only)."""
    name = "linear_dinov3_7b"
    backbone_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    backbone_indices = [39]
    scales = [1.0]
    out_channels = 1024
    num_register_tokens = 4
