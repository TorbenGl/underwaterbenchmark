"""
DINOv2 and DINOv3 model presets for semantic segmentation.
"""

from presets.Models import ModelPreset, register_model


# =============================================================================
# UPERNET + DINOV3 PRESETS
# =============================================================================

@register_model
class UperNetDinoV3Small(ModelPreset):
    """UperNet with DINOv3-Small backbone (facebook/dinov3-vits16-pretrain-lvd1689m).

    Architecture: 21M params, 12 layers, embed_dim=384, 6 heads, 4 register tokens.
    """
    name = "upernet_dinov3_small"
    backbone_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    backbone_indices = [2, 5, 8, 11]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 384
    num_register_tokens = 4


@register_model
class UperNetDinoV3Base(ModelPreset):
    """UperNet with DINOv3-Base backbone (facebook/dinov3-vitb16-pretrain-lvd1689m).

    Architecture: 86M params, 12 layers, embed_dim=768, 12 heads, 4 register tokens.
    """
    name = "upernet_dinov3_base"
    backbone_name = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    backbone_indices = [2, 5, 8, 11]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 512
    num_register_tokens = 4


@register_model
class UperNetDinoV3Large(ModelPreset):
    """UperNet with DINOv3-Large backbone (facebook/dinov3-vitl16-pretrain-lvd1689m).

    Architecture: 300M params, 24 layers, embed_dim=1024, 16 heads, 4 register tokens.
    """
    name = "upernet_dinov3_large"
    backbone_name = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768
    num_register_tokens = 4


@register_model
class UperNetDinoV3_7B(ModelPreset):
    """UperNet with DINOv3-7B backbone (facebook/dinov3-vit7b16-pretrain-lvd1689m).

    Architecture: 6.7B params, 40 layers, embed_dim=4096, 32 heads, 4 register tokens, SwiGLU FFN.
    This is the largest DINOv3 model with state-of-the-art performance.
    """
    name = "upernet_dinov3_7b"
    backbone_name = "facebook/dinov3-vit7b16-pretrain-lvd1689m"
    backbone_indices = [9, 19, 29, 39]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 1024
    num_register_tokens = 4


# =============================================================================
# LAST-LAYER-ONLY VARIANTS
# =============================================================================
# These variants use only the last transformer layer (repeated 4x) as feature
# source. The same features are projected through separate lateral convolutions
# and scaled to different spatial resolutions, creating a multi-scale pyramid
# from a single representation layer.

@register_model
class UperNetDinoV3SmallLastLayer(UperNetDinoV3Small):
    """UperNet with DINOv3-Small backbone — last-layer-only features."""
    name = "upernet_dinov3_small_lastlayer"
    backbone_indices = [11, 11, 11, 11]


@register_model
class UperNetDinoV3BaseLastLayer(UperNetDinoV3Base):
    """UperNet with DINOv3-Base backbone — last-layer-only features."""
    name = "upernet_dinov3_base_lastlayer"
    backbone_indices = [11, 11, 11, 11]


@register_model
class UperNetDinoV3LargeLastLayer(UperNetDinoV3Large):
    """UperNet with DINOv3-Large backbone — last-layer-only features."""
    name = "upernet_dinov3_large_lastlayer"
    backbone_indices = [23, 23, 23, 23]


@register_model
class UperNetDinoV3_7BLastLayer(UperNetDinoV3_7B):
    """UperNet with DINOv3-7B backbone — last-layer-only features."""
    name = "upernet_dinov3_7b_lastlayer"
    backbone_indices = [39, 39, 39, 39]
