"""
DINOv2 and DINOv3 model presets for semantic segmentation.
"""

from presets.Models import ModelPreset, register_model


# =============================================================================
# UPERNET + DINOV2 PRESETS
# =============================================================================

@register_model
class UperNetDinoV2Base(ModelPreset):
    """UperNet with DINOv2-Base backbone (facebook/dinov2-base)."""
    name = "upernet_dinov2_base"
    backbone_name = "facebook/dinov2-base"
    backbone_indices = [2, 5, 8, 11]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 512
    num_register_tokens = 0


@register_model
class UperNetDinoV2Large(ModelPreset):
    """UperNet with DINOv2-Large backbone (facebook/dinov2-large)."""
    name = "upernet_dinov2_large"
    backbone_name = "facebook/dinov2-large"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768
    num_register_tokens = 0


@register_model
class UperNetDinoV2Giant(ModelPreset):
    """UperNet with DINOv2-Giant backbone (facebook/dinov2-giant)."""
    name = "upernet_dinov2_giant"
    backbone_name = "facebook/dinov2-giant"
    backbone_indices = [9, 19, 29, 39]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 1024
    num_register_tokens = 0


@register_model
class UperNetDinoV2BaseReg(ModelPreset):
    """UperNet with DINOv2-Base + registers backbone."""
    name = "upernet_dinov2_base_reg"
    backbone_name = "facebook/dinov2-base-reg"
    backbone_indices = [2, 5, 8, 11]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 512
    num_register_tokens = 4


@register_model
class UperNetDinoV2LargeReg(ModelPreset):
    """UperNet with DINOv2-Large + registers backbone."""
    name = "upernet_dinov2_large_reg"
    backbone_name = "facebook/dinov2-large-reg"
    backbone_indices = [5, 11, 17, 23]
    scales = [4.0, 2.0, 1.0, 0.5]
    out_channels = 768
    num_register_tokens = 4


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
