"""
DINOv3 classification presets.

CLS token linear probing: frozen DINOv3 backbone + linear classifier on the
CLS token from the last transformer layer.
"""

from typing import List
import torch.nn as nn

from presets.Models import register_model
from models.backbones.classification_adapter import ClassificationBackboneAdapter
from models.classification_model import ClassificationModel


class ClsDinoPreset:
    """Base preset class for classification with DINO backbones."""

    name: str = "cls_dinov3_base"
    backbone_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    backbone_index: int = 11
    hidden_size: int = 768
    num_register_tokens: int = 4
    pooling_mode: str = "cls"

    @classmethod
    def create(
        cls,
        num_classes: int,
        img_size: tuple[int, int],
        **kwargs,
    ) -> nn.Module:
        backbone = ClassificationBackboneAdapter.from_dino(
            model_name=cls.backbone_name,
            backbone_index=cls.backbone_index,
            pooling_mode=cls.pooling_mode,
            num_register_tokens=cls.num_register_tokens,
            img_size=img_size[0],
        )

        model = ClassificationModel(
            backbone=backbone,
            num_classes=num_classes,
            hidden_size=cls.hidden_size,
        )

        return model

    @classmethod
    def get_info(cls):
        return {
            "name": cls.name,
            "backbone": cls.backbone_name,
            "backbone_index": cls.backbone_index,
            "hidden_size": cls.hidden_size,
            "pooling_mode": cls.pooling_mode,
            "num_register_tokens": cls.num_register_tokens,
        }


@register_model
class ClsDinoV3Small(ClsDinoPreset):
    """Classification with DINOv3-Small backbone (CLS token, last layer)."""
    name = "cls_dinov3_small"
    backbone_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    backbone_index = 11
    hidden_size = 384
    num_register_tokens = 4


@register_model
class ClsDinoV3Base(ClsDinoPreset):
    """Classification with DINOv3-Base backbone (CLS token, last layer)."""
    name = "cls_dinov3_base"
    backbone_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    backbone_index = 11
    hidden_size = 768
    num_register_tokens = 4


@register_model
class ClsDinoV3Large(ClsDinoPreset):
    """Classification with DINOv3-Large backbone (CLS token, last layer)."""
    name = "cls_dinov3_large"
    backbone_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    backbone_index = 23
    hidden_size = 1024
    num_register_tokens = 4


@register_model
class ClsDinoV3_7B(ClsDinoPreset):
    """Classification with DINOv3-7B backbone (CLS token, last layer)."""
    name = "cls_dinov3_7b"
    backbone_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    backbone_index = 39
    hidden_size = 4096
    num_register_tokens = 4
