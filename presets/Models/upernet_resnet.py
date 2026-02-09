"""
ResNet model presets for semantic segmentation.

Provides ImageNet-pretrained ResNet backbones as CNN baselines
alongside the ViT-based backbones (DINO, Franca, RADIO).
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones.resnet_adapter import ResNetBackboneAdapter
from models.upernet_model import UperNetSegmentationModel


class ResNetModelPreset:
    """
    Base preset class for ResNet models.

    Uses ResNetBackboneAdapter.from_timm() to create the backbone.
    """

    name: str = "upernet_resnet_base"
    model_name: str = "resnet50"
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
        backbone = ResNetBackboneAdapter.from_timm(
            model_name=cls.model_name,
            out_channels=cls.out_channels,
            pretrained=True,
        )

        num_stages = len(backbone.lateral_convs)
        hidden_sizes = [cls.out_channels] * num_stages

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
        return {
            "name": cls.name,
            "backbone": cls.model_name,
            "out_channels": cls.out_channels,
        }


@register_model
class UperNetResNet50(ResNetModelPreset):
    """UperNet with ResNet-50 backbone (ImageNet-pretrained, ~25M params)."""
    name = "upernet_resnet50"
    model_name = "resnet50"
    out_channels = 512


@register_model
class UperNetResNet101(ResNetModelPreset):
    """UperNet with ResNet-101 backbone (ImageNet-pretrained, ~44M params)."""
    name = "upernet_resnet101"
    model_name = "resnet101"
    out_channels = 512
