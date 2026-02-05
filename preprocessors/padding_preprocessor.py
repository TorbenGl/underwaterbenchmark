"""
Padding Preprocessor for semantic segmentation.

Handles images with varying aspect ratios by padding to a target size.
Provides inverse transformation to remove padding for inference.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
from PIL import Image
import numpy as np


class PaddingPreprocessor:
    """
    Preprocessor that pads images/masks to a fixed size while preserving aspect ratio.

    Features:
    - Resizes to fit within target size while maintaining aspect ratio
    - Pads to exact target size (padding on right and bottom only)
    - Tracks padding info for inverse transformation
    - Returns dict with "pixel_values" and "labels" keys

    Usage:
        preprocessor = PaddingPreprocessor(target_size=(512, 512))

        # Forward transform
        result = preprocessor(images=[img], masks=[mask])
        # result = {"pixel_values": tensor, "labels": tensor, "padding_info": [...]}

        # Inverse transform (remove padding from predictions)
        original_size_pred = preprocessor.inverse(pred_mask, padding_info)
    """

    def __init__(
        self,
        target_size: tuple[int, int] = (512, 512),
        pad_value: float = 0.0,
        mask_pad_value: int = 255,
        normalize: bool = True,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        """
        Args:
            target_size: Target (height, width) for output tensors
            pad_value: Value to use for image padding (after normalization)
            mask_pad_value: Value to use for mask padding (typically ignore_index)
            normalize: Whether to normalize images with ImageNet stats
            mean: Normalization mean (RGB)
            std: Normalization std (RGB)
        """
        self.target_size = target_size  # (H, W)
        self.pad_value = pad_value
        self.mask_pad_value = mask_pad_value
        self.normalize = normalize
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def _to_tensor(self, image: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert image to tensor [C, H, W] with values in [0, 1]."""
        if isinstance(image, torch.Tensor):
            if image.ndim == 2:
                return image.unsqueeze(0).float()
            elif image.ndim == 3 and image.shape[0] not in (1, 3):
                # HWC -> CHW
                return image.permute(2, 0, 1).float()
            return image.float()

        if isinstance(image, Image.Image):
            image = np.array(image)

        if image.ndim == 2:
            # Grayscale
            return torch.from_numpy(image).unsqueeze(0).float()
        elif image.ndim == 3:
            # HWC -> CHW
            return torch.from_numpy(image).permute(2, 0, 1).float()

        raise ValueError(f"Unexpected image shape: {image.shape}")

    def _mask_to_tensor(self, mask: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Convert mask to tensor [H, W] with integer class indices."""
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 3:
                mask = mask.squeeze(0)
            return mask.long()

        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        if mask.ndim == 3:
            # Take first channel if RGB
            mask = mask[:, :, 0]

        return torch.from_numpy(mask).long()

    def _compute_resize_and_padding(
        self,
        src_height: int,
        src_width: int,
    ) -> dict:
        """
        Compute resize dimensions and padding to fit image into target size.

        Returns dict with:
            - resize_height, resize_width: Size after resizing (before padding)
            - pad_top, pad_bottom, pad_left, pad_right: Padding amounts
            - original_height, original_width: Original dimensions
        """
        target_h, target_w = self.target_size

        # Compute scale to fit within target while preserving aspect ratio
        scale_h = target_h / src_height
        scale_w = target_w / src_width
        scale = min(scale_h, scale_w)

        # Compute resized dimensions
        resize_h = int(src_height * scale)
        resize_w = int(src_width * scale)

        # Compute padding (bottom-right only)
        pad_h = target_h - resize_h
        pad_w = target_w - resize_w

        pad_top = 0
        pad_bottom = pad_h
        pad_left = 0
        pad_right = pad_w

        return {
            "original_height": src_height,
            "original_width": src_width,
            "resize_height": resize_h,
            "resize_width": resize_w,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "scale": scale,
        }

    def _resize_tensor(
        self,
        tensor: torch.Tensor,
        size: tuple[int, int],
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """Resize tensor to target size."""
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            squeeze = 2
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
            squeeze = 1
        else:
            squeeze = 0

        if mode == "nearest":
            resized = F.interpolate(tensor.float(), size=size, mode="nearest")
        else:
            resized = F.interpolate(tensor.float(), size=size, mode=mode, align_corners=False)

        if squeeze == 2:
            return resized.squeeze(0).squeeze(0)
        elif squeeze == 1:
            return resized.squeeze(0)
        return resized

    def _pad_tensor(
        self,
        tensor: torch.Tensor,
        padding_info: dict,
        pad_value: float,
    ) -> torch.Tensor:
        """Apply padding to tensor."""
        # F.pad expects (left, right, top, bottom) for 2D/3D tensors
        pad = (
            padding_info["pad_left"],
            padding_info["pad_right"],
            padding_info["pad_top"],
            padding_info["pad_bottom"],
        )

        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
            padded = F.pad(tensor, pad, mode="constant", value=pad_value)
            return padded.squeeze(0)
        else:
            return F.pad(tensor, pad, mode="constant", value=pad_value)

    def process_single(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None,
    ) -> dict:
        """
        Process a single image (and optionally mask).

        Args:
            image: Input image (PIL, numpy, or tensor)
            mask: Optional segmentation mask

        Returns:
            Dict with "pixel_values", "labels" (if mask provided), and "padding_info"
        """
        # Convert to tensor
        img_tensor = self._to_tensor(image)

        # Normalize to [0, 1] if needed
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        # Get original size
        _, src_h, src_w = img_tensor.shape

        # Compute resize and padding
        padding_info = self._compute_resize_and_padding(src_h, src_w)

        # Resize image
        resize_size = (padding_info["resize_height"], padding_info["resize_width"])
        img_resized = self._resize_tensor(img_tensor, resize_size, mode="bilinear")

        # Apply normalization
        if self.normalize:
            img_resized = (img_resized - self.mean) / self.std

        # Pad image
        img_padded = self._pad_tensor(img_resized, padding_info, self.pad_value)

        result = {
            "pixel_values": img_padded,
            "padding_info": padding_info,
        }

        # Process mask if provided
        if mask is not None:
            mask_tensor = self._mask_to_tensor(mask)
            mask_resized = self._resize_tensor(mask_tensor, resize_size, mode="nearest").long()
            mask_padded = self._pad_tensor(mask_resized, padding_info, self.mask_pad_value).long()
            result["labels"] = mask_padded

        return result

    def __call__(
        self,
        images: list,
        masks: Optional[list] = None,
        segmentation_maps: Optional[list] = None,
        return_tensors: str = "pt",
        **kwargs,
    ) -> dict:
        """
        Process a batch of images and masks.

        Args:
            images: List of images (PIL, numpy, or tensor)
            masks: Optional list of masks
            segmentation_maps: Alias for masks (HuggingFace compatibility)
            return_tensors: Output format ("pt" for PyTorch)
            **kwargs: Ignored (for compatibility with HuggingFace preprocessors)

        Returns:
            Dict with:
                - "pixel_values": Batched image tensor [B, C, H, W]
                - "labels": Batched mask tensor [B, H, W] (if masks provided)
                - "padding_info": List of padding info dicts for each sample
        """
        # Support both 'masks' and 'segmentation_maps' parameter names
        if masks is None and segmentation_maps is not None:
            masks = segmentation_maps

        batch_images = []
        batch_masks = []
        batch_padding_info = []

        for i, image in enumerate(images):
            mask = masks[i] if masks is not None else None
            result = self.process_single(image, mask)

            batch_images.append(result["pixel_values"])
            batch_padding_info.append(result["padding_info"])

            if "labels" in result:
                batch_masks.append(result["labels"])

        output = {
            "pixel_values": torch.stack(batch_images, dim=0),
            "padding_info": batch_padding_info,
        }

        if batch_masks:
            output["labels"] = torch.stack(batch_masks, dim=0)

        return output

    def inverse(
        self,
        tensor: torch.Tensor,
        padding_info: Optional[Union[dict, list[dict]]] = None,
        mode: str = "nearest",
        denormalize: bool = True,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        Inverse transformation: optionally remove padding/resize and denormalize.

        Args:
            tensor: Prediction tensor [B, C, H, W] or [B, H, W] or [H, W]
            padding_info: Padding info dict or list of dicts (one per batch item).
                          If None, only denormalization is applied (size unchanged).
            mode: Interpolation mode for resizing (used when padding_info provided)
            denormalize: Whether to apply denormalization (for image tensors)

        Returns:
            Tensor(s) - if padding_info provided, restored to original size(s);
                        otherwise, same size with denormalization applied
        """
        # Denormalize if requested (for image tensors with channels)
        if denormalize and tensor.ndim >= 3:
            # Check if this looks like an image tensor (has 3 channels)
            if (tensor.ndim == 3 and tensor.shape[0] == 3) or \
               (tensor.ndim == 4 and tensor.shape[1] == 3):
                tensor = self._denormalize(tensor)

        # If no padding_info, just return (possibly denormalized) tensor
        if padding_info is None:
            return tensor

        # Handle single tensor (no batch dim)
        if isinstance(padding_info, dict):
            return self._inverse_single(tensor, padding_info, mode)

        # Handle batched tensor
        results = []
        for i, info in enumerate(padding_info):
            if tensor.ndim == 4:
                single = tensor[i]  # [C, H, W]
            elif tensor.ndim == 3:
                single = tensor[i]  # [H, W]
            else:
                raise ValueError(f"Unexpected tensor shape for batch: {tensor.shape}")

            results.append(self._inverse_single(single, info, mode))

        return results

    def _inverse_single(
        self,
        tensor: torch.Tensor,
        padding_info: dict,
        mode: str = "nearest",
    ) -> torch.Tensor:
        """Remove padding and resize single tensor back to original size."""
        # Remove padding
        pad_top = padding_info["pad_top"]
        pad_bottom = padding_info["pad_bottom"]
        pad_left = padding_info["pad_left"]
        pad_right = padding_info["pad_right"]

        h, w = tensor.shape[-2:]

        # Compute crop indices
        top = pad_top
        bottom = h - pad_bottom if pad_bottom > 0 else h
        left = pad_left
        right = w - pad_right if pad_right > 0 else w

        # Crop out padding
        if tensor.ndim == 2:
            cropped = tensor[top:bottom, left:right]
        elif tensor.ndim == 3:
            cropped = tensor[:, top:bottom, left:right]
        else:
            cropped = tensor[:, :, top:bottom, left:right]

        # Resize back to original dimensions
        original_size = (padding_info["original_height"], padding_info["original_width"])
        restored = self._resize_tensor(cropped, original_size, mode=mode)

        return restored

    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Internal denormalization helper."""
        if not self.normalize:
            return tensor

        mean = self.mean.to(tensor.device)
        std = self.std.to(tensor.device)

        if tensor.ndim == 4:
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)

        return tensor * std + mean

    def inverse_denormalize(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Denormalize image tensor (undo ImageNet normalization).

        Alias for backwards compatibility. Prefer using inverse() with denormalize=True.
        """
        return self._denormalize(tensor)
