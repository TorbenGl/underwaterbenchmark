"""
Preprocessors for semantic segmentation training.

Usage:
    from preprocessors import PaddingPreprocessor

    preprocessor = PaddingPreprocessor(target_size=(512, 512))

    # Forward transform
    result = preprocessor(images=[img], masks=[mask])
    # result = {"pixel_values": tensor, "labels": tensor, "padding_info": [...]}

    # Inverse transform (remove padding from predictions)
    original_size_pred = preprocessor.inverse(pred_mask, padding_info)
"""

from preprocessors.padding_preprocessor import PaddingPreprocessor

__all__ = ["PaddingPreprocessor"]
