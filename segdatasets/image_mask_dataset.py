from PIL import Image
from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from typing import Optional, Sequence
import numpy as np


class ImageMaskDataset(Dataset):
    """
    Dataset for semantic segmentation from lists of image and mask paths
    or from an annotation file.

    Two modes of initialization:
        1. Direct paths: Provide image_paths and mask_paths lists
        2. Annotation file: Provide annotation_file path (JSON with image/mask pairs)

    Supports RGB color-coded masks: If rgb_mask=True, the dataset will auto-detect
    unique RGB colors and map them to sequential class indices.

    All paths are relative to the root directory.

    Label contract (STRICT):
        0  -> background
        1  -> first category
        ...
        N  -> Nth category

    Returns:
        image: FloatTensor [3, H, W] in [0, 1]
        mask:  LongTensor  [H, W] with values in [0, N]
    """

    def __init__(
        self,
        root: str | Path,
        image_paths: Sequence[str | Path] | None = None,
        mask_paths: Sequence[str | Path] | None = None,
        annotation_file: str | Path | None = None,
        transforms: T.Compose | None = None,
        id2label: Optional[dict[int, str]] = None,
        num_classes: Optional[int] = None,
        cache_dataset: bool = False,
        cached_images: Optional[torch.Tensor] = None,
        cached_masks: Optional[torch.Tensor] = None,
        rgb_mask: bool = False,
        color_to_class: Optional[dict[tuple[int, int, int], int]] = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
        increase_idx: bool = False,
        value_to_class: Optional[dict[int, int]] = None,
    ):
        """
        Args:
            root: Root directory for all paths.
            image_paths: List of paths to RGB images (relative to root).
            mask_paths: List of paths to mask images (relative to root).
            annotation_file: Path to JSON annotation file (relative to root).
                             Expected format:
                             {
                                 "samples": [
                                     {"image": "path/to/img.jpg", "mask": "path/to/mask.png"},
                                     ...
                                 ]
                             }
            transforms: Optional torchvision transforms to apply.
            id2label: Optional mapping from class id to label name.
            num_classes: Number of classes (including background). If None,
                         inferred from masks on first access.
            cache_dataset: If True, use cached tensors instead of loading from disk.
            cached_images: Pre-built tensor of cached images [N, 3, H, W] for shared memory.
            cached_masks: Pre-built tensor of cached masks [N, H, W] for shared memory.
            rgb_mask: If True, masks are RGB color-coded and will be converted to class indices.
            color_to_class: Optional explicit mapping from RGB tuple to class index.
                            If None and rgb_mask=True, colors are auto-detected.
            background_color: RGB color to treat as background (class 0). Default: (0, 0, 0).
            increase_idx: If True, normalize mask values so that 0=background and
                          non-zero values become sequential class indices starting at 1.
                          Useful for binary masks (0/255 -> 0/1).
            value_to_class: Optional explicit mapping from grayscale mask values to class indices.
                            Example: {0: 0, 255: 1} maps pixel value 0 to class 0, 255 to class 1.
                            Takes precedence over increase_idx when provided.
        """
        self.root = Path(root)
        self.transforms = transforms
        self.id2label = id2label or {}
        self._num_classes = num_classes
        self.cache_dataset = cache_dataset
        self.rgb_mask = rgb_mask
        self.background_color = background_color
        self.increase_idx = increase_idx
        self.value_to_class = value_to_class

        # Load paths from either direct lists or annotation file
        if annotation_file is not None:
            # Mode 2: Load from annotation file
            if image_paths is not None or mask_paths is not None:
                raise ValueError(
                    "Cannot use both annotation_file and image_paths/mask_paths. "
                    "Choose one method."
                )
            self.image_paths, self.mask_paths = self._load_annotation_file(
                annotation_file
            )
        elif image_paths is not None and mask_paths is not None:
            # Mode 1: Direct paths
            if len(image_paths) != len(mask_paths):
                raise ValueError(
                    f"Number of images ({len(image_paths)}) must match "
                    f"number of masks ({len(mask_paths)})"
                )
            self.image_paths = [self.root / Path(p) for p in image_paths]
            self.mask_paths = [self.root / Path(p) for p in mask_paths]
        else:
            raise ValueError(
                "Must provide either (image_paths and mask_paths) or annotation_file"
            )

        # RGB color to class mapping
        self.color_to_class: Optional[dict[tuple[int, int, int], int]] = color_to_class
        if self.rgb_mask and self.color_to_class is None:
            self.color_to_class = self._auto_detect_colors()

        # Optional cache (shared memory tensors)
        self._image_cache = cached_images
        self._mask_cache = cached_masks

    def _load_annotation_file(
        self, annotation_file: str | Path
    ) -> tuple[list[Path], list[Path]]:
        """Load image and mask paths from a JSON annotation file."""
        ann_path = self.root / annotation_file
        with open(ann_path, "r") as f:
            data = json.load(f)

        image_paths = []
        mask_paths = []

        for sample in data["samples"]:
            image_paths.append(self.root / sample["image"])
            mask_paths.append(self.root / sample["mask"])

        return image_paths, mask_paths

    def _auto_detect_colors(self) -> dict[tuple[int, int, int], int]:
        """Scan all masks to detect unique RGB colors and assign class indices."""
        unique_colors: set[tuple[int, int, int]] = set()

        for mask_path in self.mask_paths:
            mask_img = Image.open(mask_path).convert("RGB")
            mask_array = np.array(mask_img)
            # Reshape to (N, 3) where N is number of pixels
            pixels = mask_array.reshape(-1, 3)
            # Get unique colors
            colors = set(map(tuple, pixels))
            unique_colors.update(colors)

        # Build mapping: background_color -> 0, other colors -> 1, 2, 3, ...
        color_to_class = {}
        if self.background_color in unique_colors:
            color_to_class[self.background_color] = 0
            unique_colors.remove(self.background_color)

        # Sort remaining colors for deterministic ordering
        sorted_colors = sorted(unique_colors)
        for idx, color in enumerate(sorted_colors, start=1):
            color_to_class[color] = idx

        # Update num_classes based on detected colors
        self._num_classes = len(color_to_class)

        return color_to_class

    def _rgb_to_class_mask(self, mask_img: Image.Image) -> np.ndarray:
        """Convert RGB mask to class index mask using color_to_class mapping."""
        mask_rgb = np.array(mask_img.convert("RGB"))
        h, w = mask_rgb.shape[:2]
        mask_class = np.zeros((h, w), dtype=np.int64)

        for color, class_idx in self.color_to_class.items():
            # Create boolean mask where all 3 channels match the color
            matches = np.all(mask_rgb == np.array(color), axis=2)
            mask_class[matches] = class_idx

        return mask_class

    def _load_image(self, index: int) -> tv_tensors.Image:
        """Load an RGB image from disk."""
        img_path = self.image_paths[index]
        image = Image.open(img_path).convert("RGB")
        return tv_tensors.Image(image)

    def _load_mask(self, index: int) -> tv_tensors.Mask:
        """Load a mask from disk as a LongTensor."""
        mask_path = self.mask_paths[index]
        mask_img = Image.open(mask_path)

        if self.rgb_mask:
            # Convert RGB colors to class indices
            mask_array = self._rgb_to_class_mask(mask_img)
        elif mask_img.mode == "P":
            mask_array = np.array(mask_img, dtype=np.int64)
        elif mask_img.mode in ("L", "I"):
            mask_array = np.array(mask_img, dtype=np.int64)
        elif mask_img.mode == "RGB":
            mask_array = np.array(mask_img.convert("L"), dtype=np.int64)
        else:
            mask_array = np.array(mask_img.convert("L"), dtype=np.int64)

        # Apply value_to_class mapping if provided (takes precedence over increase_idx)
        if self.value_to_class is not None:
            mapped_mask = np.zeros_like(mask_array, dtype=np.int64)
            for src_value, dst_class in self.value_to_class.items():
                mapped_mask[mask_array == src_value] = dst_class
            mask_array = mapped_mask
        # Normalize mask values if increase_idx is enabled
        # Maps 0 -> 0 (background), non-zero -> 1 (for binary masks like 0/255)
        elif self.increase_idx:
            mask_array = (mask_array > 0).astype(np.int64)

        mask = torch.from_numpy(mask_array).long()
        return tv_tensors.Mask(mask)

    def _load_sample(self, index: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Load an image-mask pair from disk."""
        image = self._load_image(index)
        mask = self._load_mask(index)
        return image, mask

    @property
    def num_classes(self) -> int:
        """Return the number of classes (inferred from masks if not provided)."""
        if self._num_classes is not None:
            return self._num_classes

        if self._mask_cache is not None:
            max_val = self._mask_cache.max().item()
        else:
            max_val = 0
            for idx in range(len(self.mask_paths)):
                mask = self._load_mask(idx)
                max_val = max(max_val, mask.max().item())

        self._num_classes = int(max_val) + 1
        return self._num_classes

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get item from shared cache or load from disk."""
        if self.cache_dataset:
            # Load from shared memory
            image = self._image_cache[index].clone()  # Clone to avoid modifying shared data
            mask = self._mask_cache[index].clone()
        else:
            # Load from disk
            image, mask = self._load_sample(index)
        # Apply transforms (e.g., augmentation for training)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image, mask

    def __len__(self) -> int:
        return len(self.image_paths)
