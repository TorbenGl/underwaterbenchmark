"""
Lightning DataModule for ImageMaskDataset (simple image/mask pairs).

Supports:
- Direct image/mask path lists
- JSON annotation files
- RGB color-coded masks
- Shared memory caching
"""

import lightning
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Optional, Sequence, Dict

from segdatasets.image_mask_dataset import ImageMaskDataset


class ImageMaskDataModule(lightning.LightningDataModule):
    """
    Lightning DataModule for ImageMaskDataset.

    Supports two modes:
    1. Annotation files: JSON files with {"samples": [{"image": ..., "mask": ...}, ...]}
    2. Direct paths: Lists of image and mask paths

    Features:
    - Shared memory caching for efficient distributed training
    - RGB color-coded mask support
    - HuggingFace preprocessor integration
    """

    def __init__(
        self,
        root: str,
        # Annotation file mode
        train_annotation_file: Optional[str] = None,
        val_annotation_file: Optional[str] = None,
        test_annotation_file: Optional[str] = None,
        # Direct paths mode
        train_image_paths: Optional[Sequence[str]] = None,
        train_mask_paths: Optional[Sequence[str]] = None,
        val_image_paths: Optional[Sequence[str]] = None,
        val_mask_paths: Optional[Sequence[str]] = None,
        test_image_paths: Optional[Sequence[str]] = None,
        test_mask_paths: Optional[Sequence[str]] = None,
        # Common settings
        batch_size: int = 8,
        num_workers: int = 4,
        img_size: tuple[int, int] = (512, 512),
        preprocessor=None,
        num_classes: Optional[int] = None,
        id2label: Optional[dict[int, str]] = None,
        # RGB mask settings
        rgb_mask: bool = False,
        color_to_class: Optional[dict[tuple[int, int, int], int]] = None,
        background_color: tuple[int, int, int] = (0, 0, 0),
        # Mask normalization
        increase_idx: bool = False,
        value_to_class: Optional[dict[int, int]] = None,
        # DataLoader settings
        devices: str | list = "auto",
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cache_dataset: bool = False,
        prefetch_factor: int = 2,
    ) -> None:
        super().__init__()

        # Validate img_size when caching
        if cache_dataset and img_size is None:
            raise ValueError(
                "img_size must be specified when cache_dataset=True."
            )

        # Core parameters
        self.root = Path(root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.preprocessor = preprocessor
        self._num_classes = num_classes
        self.id2label = id2label
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.cache_dataset = cache_dataset

        # RGB mask settings
        self.rgb_mask = rgb_mask
        self.color_to_class = color_to_class
        self.background_color = background_color
        self.increase_idx = increase_idx
        self.value_to_class = value_to_class

        # Annotation files
        self.train_annotation_file = train_annotation_file
        self.val_annotation_file = val_annotation_file
        self.test_annotation_file = test_annotation_file

        # Direct paths
        self.train_image_paths = train_image_paths
        self.train_mask_paths = train_mask_paths
        self.val_image_paths = val_image_paths
        self.val_mask_paths = val_mask_paths
        self.test_image_paths = test_image_paths
        self.test_mask_paths = test_mask_paths

        # Parse devices
        self.devices = (
            devices if devices == "auto"
            else _parse_gpu_ids(devices, include_cuda=True)
        )

        # Datasets (created in setup)
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Shared memory tensors
        self.shared_train_images = None
        self.shared_train_masks = None
        self.shared_val_images = None
        self.shared_val_masks = None
        self.shared_test_images = None
        self.shared_test_masks = None

    def _build_shared_cache(
        self,
        annotation_file: Optional[str],
        image_paths: Optional[Sequence[str]],
        mask_paths: Optional[Sequence[str]],
        split_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build shared memory cache with preprocessed data."""
        print(f"Building shared memory cache for {split_name} split...")

        # Create temporary dataset for loading
        temp_dataset = ImageMaskDataset(
            root=self.root,
            annotation_file=annotation_file,
            image_paths=image_paths,
            mask_paths=mask_paths,
            transforms=None,
            rgb_mask=self.rgb_mask,
            color_to_class=self.color_to_class,
            background_color=self.background_color,
            increase_idx=self.increase_idx,
            value_to_class=self.value_to_class,
        )

        num_samples = len(temp_dataset)
        h, w = self.img_size

        shared_images = torch.zeros(
            (num_samples, 3, h, w),
            dtype=torch.float32
        ).share_memory_()

        shared_masks = torch.zeros(
            (num_samples, h, w),
            dtype=torch.long
        ).share_memory_()

        # Load, preprocess, and cache all samples
        for idx in range(num_samples):
            image, mask = temp_dataset._load_sample(idx)

            # Apply preprocessing
            if self.preprocessor is not None:
                processed = self.preprocessor(
                    images=[image],
                    return_tensors="pt",
                    do_resize=True,
                    size={"height": self.img_size[0], "width": self.img_size[1]},
                )
                preprocessed_image = processed["pixel_values"][0]
                preprocessed_mask = T.functional.resize(
                    mask.unsqueeze(0),
                    size=self.img_size,
                    interpolation=T.InterpolationMode.NEAREST,
                ).squeeze(0)
            else:
                preprocessed_image = T.functional.resize(
                    image,
                    size=self.img_size,
                    interpolation=T.InterpolationMode.BILINEAR,
                )
                preprocessed_mask = T.functional.resize(
                    mask.unsqueeze(0),
                    size=self.img_size,
                    interpolation=T.InterpolationMode.NEAREST,
                ).squeeze(0)

            shared_images[idx] = preprocessed_image
            shared_masks[idx] = preprocessed_mask

            if (idx + 1) % 100 == 0:
                print(f"  Cached {idx + 1}/{num_samples} samples")

        print(f"Caching completed for {split_name}.")
        return shared_images, shared_masks

    def _create_dataset(
        self,
        annotation_file: Optional[str],
        image_paths: Optional[Sequence[str]],
        mask_paths: Optional[Sequence[str]],
        cached_images: Optional[torch.Tensor] = None,
        cached_masks: Optional[torch.Tensor] = None,
    ) -> ImageMaskDataset:
        """Create an ImageMaskDataset instance."""
        return ImageMaskDataset(
            root=self.root,
            annotation_file=annotation_file,
            image_paths=image_paths,
            mask_paths=mask_paths,
            transforms=None,
            id2label=self.id2label,
            num_classes=self._num_classes,
            cache_dataset=self.cache_dataset,
            cached_images=cached_images,
            cached_masks=cached_masks,
            rgb_mask=self.rgb_mask,
            color_to_class=self.color_to_class,
            background_color=self.background_color,
            increase_idx=self.increase_idx,            
            value_to_class=self.value_to_class,
        )

    def prepare_data(self):
        """Download/prepare data if needed."""
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup datasets for each stage."""
        if stage == "fit" or stage is None:
            # Validation dataset
            if self.val_annotation_file or self.val_image_paths:
                if self.cache_dataset:
                    self.shared_val_images, self.shared_val_masks = self._build_shared_cache(
                        self.val_annotation_file,
                        self.val_image_paths,
                        self.val_mask_paths,
                        "val"
                    )
                self.val_dataset = self._create_dataset(
                    self.val_annotation_file,
                    self.val_image_paths,
                    self.val_mask_paths,
                    self.shared_val_images,
                    self.shared_val_masks,
                )

            # Training dataset
            if self.train_annotation_file or self.train_image_paths:
                if self.cache_dataset:
                    self.shared_train_images, self.shared_train_masks = self._build_shared_cache(
                        self.train_annotation_file,
                        self.train_image_paths,
                        self.train_mask_paths,
                        "train"
                    )
                self.train_dataset = self._create_dataset(
                    self.train_annotation_file,
                    self.train_image_paths,
                    self.train_mask_paths,
                    self.shared_train_images,
                    self.shared_train_masks,
                )

            # Store color_to_class from dataset if auto-detected
            if self.rgb_mask and self.color_to_class is None and self.train_dataset:
                self.color_to_class = self.train_dataset.color_to_class

        if stage == "test" or stage is None:
            if self.test_annotation_file or self.test_image_paths:
                if self.cache_dataset:
                    self.shared_test_images, self.shared_test_masks = self._build_shared_cache(
                        self.test_annotation_file,
                        self.test_image_paths,
                        self.test_mask_paths,
                        "test"
                    )
                self.test_dataset = self._create_dataset(
                    self.test_annotation_file,
                    self.test_image_paths,
                    self.test_mask_paths,
                    self.shared_test_images,
                    self.shared_test_masks,
                )

    @property
    def num_classes(self) -> int:
        """Get number of classes."""
        if self._num_classes is not None:
            return self._num_classes
        if self.train_dataset is not None:
            return self.train_dataset.num_classes
        raise RuntimeError("Call setup() before accessing num_classes")

    def get_num_classes(self) -> int:
        """Get number of classes (alias for compatibility)."""
        return self.num_classes

    def get_id2label(self) -> Dict[int, str]:
        """Get mapping from class IDs to labels."""
        if self.id2label is None:
            raise RuntimeError("id2label not set. Provide id2label in constructor or check dataset.")
        return self.id2label

    def collate_fn(self, batch):
        """
        Collate function to batch samples.

        Handles variable-sized images by processing each sample individually
        before stacking (when not using cache).
        """
        images, masks = zip(*batch)

        # When using cache: just stack (uniform size)
        if self.cache_dataset:
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            return {
                "pixel_values": images,
                "labels": masks,
            }

        # Apply preprocessing on-the-fly if preprocessor is provided
        # Preprocessor handles variable-sized inputs
        elif self.preprocessor is not None:
            batch_dict = self.preprocessor(
                images=list(images),
                segmentation_maps=list(masks),
                return_tensors="pt",
            )
            # Normalize output keys to pixel_values, labels, and padding_info
            result = {
                "pixel_values": batch_dict["pixel_values"],
                "labels": batch_dict["labels"],
            }
            if "padding_info" in batch_dict:
                result["padding_info"] = batch_dict["padding_info"]
            return result

        # No cache and no preprocessor: resize each sample individually
        else:
            resized_images = []
            resized_masks = []
            for img, mask in zip(images, masks):
                resized_img = T.functional.resize(
                    img,
                    size=self.img_size,
                    interpolation=T.InterpolationMode.BILINEAR,
                )
                resized_mask = T.functional.resize(
                    mask.unsqueeze(0),
                    size=self.img_size,
                    interpolation=T.InterpolationMode.NEAREST,
                ).squeeze(0)
                resized_images.append(resized_img)
                resized_masks.append(resized_mask)

            return {
                "pixel_values": torch.stack(resized_images, dim=0),
                "labels": torch.stack(resized_masks, dim=0),
            }

    def train_dataloader(self) -> DataLoader:
        """Training dataloader with shuffling."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader without shuffling."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def test_dataloader(self) -> DataLoader:
        """Test dataloader without shuffling."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up shared memory resources."""
        if stage == "fit":
            if self.shared_val_images is not None:
                del self.shared_val_images
                del self.shared_val_masks
                self.shared_val_images = None
                self.shared_val_masks = None

            if self.shared_train_images is not None:
                del self.shared_train_images
                del self.shared_train_masks
                self.shared_train_images = None
                self.shared_train_masks = None

        if stage == "test":
            if self.shared_test_images is not None:
                del self.shared_test_images
                del self.shared_test_masks
                self.shared_test_images = None
                self.shared_test_masks = None
