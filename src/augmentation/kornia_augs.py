"""
Kornia-based augmentation pipeline for semantic segmentation.

Benefits:
- GPU-accelerated (much faster than CPU transforms)
- Batch-level processing (more efficient)
- Automatic image/mask synchronization
- Differentiable (can backprop through augmentations if needed)

Usage:
    aug = SemanticSegmentationAugmentation(img_size=(512, 512))
    images, masks = aug(images, masks)  # On GPU
"""
import torch
import torch.nn as nn
import kornia.augmentation as K
from kornia.geometry.transform import resize
from typing import Tuple

class SemanticSegmentationAugmentation(nn.Module):
    """
    GPU-accelerated augmentation pipeline for semantic segmentation.

    Uses Kornia's AugmentationSequential to apply the SAME geometric
    transforms to both images and masks, while only applying color
    transforms to images.

    Args:
        img_size: Target image size (H, W)
        train: If True, apply training augmentations. If False, only resize.
        p: Base probability for augmentations
        color_jitter: Apply color jittering (only to images)
        random_crop: Apply random cropping
        horizontal_flip: Apply random horizontal flips
        vertical_flip: Apply random vertical flips
        random_rotation: Apply random rotations (degrees)
        random_scale: Apply random scaling (min_scale, max_scale)
    """

    def __init__(
        self,
        img_size: Tuple[int, int] = (512, 512),
        train: bool = True,
        p: float = 0.5,
        color_jitter: bool = True,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        random_rotation: float = 0.0,  # degrees
        random_scale: Tuple[float, float] = None,  # (min, max)
        random_crop: bool = False,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.is_training = train

        if not self.is_training:
            # Validation/test: no augmentation, just ensure correct size
            self.transform = None
            return

        # Build augmentation pipeline
        # Note: Use same_on_batch=False to apply different augmentations per sample
        # Use data_keys to specify what to augment
        augmentations = []

        # === GEOMETRIC TRANSFORMS (applied to both image and mask) ===

        if horizontal_flip:
            augmentations.append(
                K.RandomHorizontalFlip(p=p, same_on_batch=False)
            )

        if vertical_flip:
            augmentations.append(
                K.RandomVerticalFlip(p=p, same_on_batch=False)
            )

        if random_rotation > 0:
            augmentations.append(
                K.RandomRotation(
                    degrees=random_rotation,
                    p=p,
                    same_on_batch=False,
                    resample="bilinear",  # For images
                    align_corners=False,
                )
            )

        if random_scale is not None:
            min_scale, max_scale = random_scale
            augmentations.append(
                K.RandomAffine(
                    degrees=0,
                    scale=(min_scale, max_scale),
                    p=p,
                    same_on_batch=False,
                    resample="bilinear",
                    align_corners=False,
                )
            )

        if random_crop:
            augmentations.append(
                K.RandomCrop(size=img_size, p=1.0, same_on_batch=False)
            )

        # === COLOR TRANSFORMS (applied only to images) ===

        if color_jitter:
            augmentations.append(
                K.ColorJitter(
                    brightness=brightness,
                    contrast=contrast,
                    saturation=saturation,
                    hue=hue,
                    p=p,
                    same_on_batch=False,
                )
            )

        # Build the augmentation container
        if augmentations:
            self.transform = K.AugmentationSequential(
                *augmentations,
                data_keys=["input", "mask"],  # Apply to both image and mask
                same_on_batch=False,
            )
        else:
            self.transform = None

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply augmentations to images and masks.

        Args:
            images: [B, C, H, W] float tensor in [0, 1] or [-1, 1]
            masks: [B, H, W] or [B, 1, H, W] long tensor

        Returns:
            images: [B, C, H, W] augmented images
            masks: [B, H, W] augmented masks (same geometric transforms)
        """
        if not self.is_training or self.transform is None:
            # No augmentation, just ensure correct size
            if images.shape[-2:] != self.img_size:
                images = resize(images, self.img_size, interpolation="bilinear", align_corners=False)
                masks = resize(
                    masks.unsqueeze(1).float(),
                    self.img_size,
                    interpolation="nearest"
                ).squeeze(1).long()
            return images, masks

        # Ensure masks have channel dimension for Kornia
        if masks.ndim == 3:
            masks_4d = masks.unsqueeze(1).float()  # [B, 1, H, W]
        else:
            masks_4d = masks.float()

        # Apply augmentations
        # Kornia expects float masks, we'll convert back to long after
        aug_images, aug_masks = self.transform(images, masks_4d)

        # Convert masks back to long and remove channel dimension
        aug_masks = aug_masks.squeeze(1).long()

        return aug_images, aug_masks


class LightAugmentation(SemanticSegmentationAugmentation):
    """Light augmentation preset for quick training."""
    def __init__(self, img_size=(512, 512)):
        super().__init__(
            img_size=img_size,
            train=True,
            p=0.5,
            color_jitter=True,
            horizontal_flip=True,
            vertical_flip=False,
            random_rotation=0.0,
            random_scale=None,
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05,
        )


class MediumAugmentation(SemanticSegmentationAugmentation):
    """Medium augmentation preset (recommended for most cases)."""
    def __init__(self, img_size=(512, 512)):
        super().__init__(
            img_size=img_size,
            train=True,
            p=0.5,
            color_jitter=True,
            horizontal_flip=True,
            vertical_flip=False,
            random_rotation=10.0,
            random_scale=(0.9, 1.1),
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        )


class HeavyAugmentation(SemanticSegmentationAugmentation):
    """Heavy augmentation preset for challenging datasets."""
    def __init__(self, img_size=(512, 512)):
        super().__init__(
            img_size=img_size,
            train=True,
            p=0.7,
            color_jitter=True,
            horizontal_flip=True,
            vertical_flip=True,
            random_rotation=20.0,
            random_scale=(0.8, 1.2),
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.15,
        )


# For underwater imagery - specialized augmentations
class UnderwaterAugmentation(SemanticSegmentationAugmentation):
    """
    Specialized augmentation for underwater imagery.

    Focuses on color shifts and lighting variations common in underwater scenes.
    """
    def __init__(self, img_size=(512, 512)):
        super().__init__(
            img_size=img_size,
            train=True,
            p=0.6,
            color_jitter=True,
            horizontal_flip=True,
            vertical_flip=False,
            random_rotation=5.0,  # Less rotation for underwater scenes
            random_scale=(0.95, 1.05),  # Subtle scaling
            brightness=0.3,  # Higher for underwater lighting variation
            contrast=0.3,
            saturation=0.4,  # Higher to simulate depth/turbidity
            hue=0.2,  # Higher for blue-green shifts
        )
