import lightning
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from torch.utils.data import DataLoader
import torch
import torchvision.transforms as T
from datasets import load_dataset
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from segdatasets.hf_semantic_dataset import HuggingFaceSemanticDataset


class HuggingFaceLightningDataModule_Semantic(lightning.LightningDataModule):
    """
    Lightning DataModule for Semantic Segmentation using HuggingFace Datasets.

    Supports both:
    - Loading from HuggingFace Hub (dataset_name="EPFL-ECEO/coralscapes")
    - Loading from local directory (data_dir="/path/to/dataset")

    Key Features:
    - Preprocessor is applied BEFORE caching (cache stores preprocessed data)
    - Shared memory caching for efficient distributed training
    - All workers access the same cached tensors without duplication
    - Separate control for caching train/val/test splits

    WARNING: Caching requires significant RAM. For large datasets at 512x512,
    expect ~50GB+ of memory usage. Ensure sufficient system resources.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        dataset_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        split_mapping: Optional[Dict[str, str]] = None,
        image_column: str = "image",
        mask_column: str = "label",
        preprocessor=None,  # Applied BEFORE caching if provided
        devices: str | list = "auto",
        ignore_idx: Optional[int] = None,
        id2label: Optional[Dict[int, str]] = None,
        num_classes: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cache_dataset: bool = False,  # Master switch for all caching
        prefetch_factor: int = 2,
        train_transforms=None,  # Applied AFTER cache retrieval (augmentations only)
        val_transforms=None,    # Applied AFTER cache retrieval
        trust_remote_code: bool = False,
    ) -> None:
        """
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            img_size: Target image size (height, width)
            dataset_name: HuggingFace dataset name (e.g., "scene_parse_150", "EPFL-ECEO/coralscapes")
                         Either dataset_name or data_dir must be provided.
            data_dir: Path for dataset storage. Behavior depends on dataset_name:
                     - If dataset_name is None: Load from local directory (HuggingFace format)
                     - If dataset_name is set: Use as cache directory for HuggingFace downloads
                     This allows datasets to be stored in a specific location for reuse.
            split_mapping: Mapping of standard splits to dataset splits
                          Default: {"train": "train", "val": "validation", "test": "test"}
            image_column: Column name for images in the dataset
            mask_column: Column name for segmentation masks
            preprocessor: HuggingFace image processor (applied before caching)
            devices: Device specification
            ignore_idx: Index to ignore in loss computation
            id2label: Optional mapping from class IDs to labels
            num_classes: Optional number of classes
            pin_memory: Whether to pin memory in dataloaders
            persistent_workers: Whether to use persistent workers
            cache_dataset: Whether to cache dataset in shared memory
            prefetch_factor: Prefetch factor for dataloaders
            train_transforms: Transforms for training (augmentations, applied after cache)
            val_transforms: Transforms for validation/test (applied after cache)
            trust_remote_code: Whether to trust remote code for dataset loading
        """
        super().__init__()

        # Validate that either dataset_name or data_dir is provided
        if dataset_name is None and data_dir is None:
            raise ValueError("Either dataset_name or data_dir must be provided")
        if dataset_name is not None and data_dir is not None:
            print("INFO: Both dataset_name and data_dir provided. Using data_dir as cache directory for HuggingFace downloads.")

        # Validate img_size when caching
        if cache_dataset and img_size is None:
            raise ValueError(
                "img_size must be specified when cache_dataset=True. "
                "This ensures consistent tensor shapes in the cache."
            )

        # Warn about preprocessor + caching interaction
        if preprocessor is not None and cache_dataset:
            print(
                "INFO: Preprocessor will be applied BEFORE caching. "
                "Cached data will contain preprocessed tensors, not raw images. "
            )

        # Core parameters
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) if data_dir is not None else None
        # Only use local loading if data_dir is provided WITHOUT dataset_name
        # If both are provided, download from Hub to data_dir as cache
        self.use_local = data_dir is not None and dataset_name is None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.preprocessor = preprocessor
        self.ignore_idx = ignore_idx
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.trust_remote_code = trust_remote_code

        # Column names
        self.image_column = image_column
        self.mask_column = mask_column

        # Class info
        self._id2label = id2label
        self._num_classes = num_classes

        # Split mapping (HuggingFace datasets use various split names)
        self.split_mapping = split_mapping or {
            "train": "train",
            "val": "validation",
            "test": "test",
        }

        # Caching configuration
        self.cache_dataset = cache_dataset

        # Parse devices (kept for potential future use)
        self.devices = (
            devices if devices == "auto"
            else _parse_gpu_ids(devices, include_cuda=True)
        )

        # Transforms (applied AFTER cache retrieval - should be augmentations only)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = None
        self.id2label = None

        # Shared memory tensors (created in setup())
        self.shared_val_images = None
        self.shared_val_masks = None
        self.shared_train_images = None
        self.shared_train_masks = None
        self.shared_test_images = None
        self.shared_test_masks = None

        # Pre-loaded HuggingFace datasets
        self._hf_datasets: Dict[str, Any] = {}

    def _load_hf_dataset(self, split_key: str) -> Any:
        """Load HuggingFace dataset for a given split (from Hub or local)."""
        if split_key not in self._hf_datasets:
            hf_split = self.split_mapping.get(split_key, split_key)
            try:
                if self.use_local:
                    # Load from local directory
                    self._hf_datasets[split_key] = load_dataset(
                        str(self.data_dir),
                        split=hf_split,
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    # Load from HuggingFace Hub
                    # Use data_dir as cache directory if provided
                    self._hf_datasets[split_key] = load_dataset(
                        self.dataset_name,
                        split=hf_split,
                        trust_remote_code=self.trust_remote_code,
                        cache_dir=str(self.data_dir) if self.data_dir is not None else None,
                    )
            except ValueError as e:
                # Handle case where split doesn't exist
                print(f"Warning: Could not load split '{hf_split}': {e}")
                return None
        return self._hf_datasets[split_key]

    def _build_shared_cache(
        self,
        split_key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build shared memory cache with PREPROCESSED data.

        Process flow:
        1. Load raw image from HuggingFace
        2. Apply preprocessor/resize
        3. Store preprocessed result in shared memory

        Returns:
            shared_images: Tensor [N, C, H, W] in shared memory (preprocessed)
            shared_masks: Tensor [N, H, W] in shared memory (preprocessed)
        """
        print(f"Building shared memory cache for {split_key} split...")
        print(f"  Preprocessing will be applied before caching")
        if self.preprocessor is not None:
            print(f"  Using preprocessor")
        else:
            print(f"  Using standard resize to {self.img_size}")

        # Create temporary dataset for loading
        hf_dataset = self._load_hf_dataset(split_key)
        temp_dataset = HuggingFaceSemanticDataset(
            dataset_name=self.dataset_name,
            split=self.split_mapping.get(split_key, split_key),
            image_column=self.image_column,
            mask_column=self.mask_column,
            transforms=None,  # No transforms during caching
            id2label=self._id2label,
            num_classes=self._num_classes,
            hf_dataset=hf_dataset,
            trust_remote_code=self.trust_remote_code,
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
            # Load raw sample
            image, mask = temp_dataset._load_sample(idx)

            # Apply preprocessing BEFORE caching
            if self.preprocessor is not None:
                # Use HuggingFace preprocessor
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

            # Store PREPROCESSED data in shared memory
            shared_images[idx] = preprocessed_image
            shared_masks[idx] = preprocessed_mask

            if (idx + 1) % 100 == 0:
                print(f"  Cached {idx + 1}/{num_samples} samples (preprocessed)")

        print(f"Caching completed for {split_key}. Data is preprocessed and ready.")
        return shared_images, shared_masks

    def prepare_data(self):
        """
        Download/prepare data if needed. Called only on rank 0 in distributed training.
        HuggingFace datasets handles downloading automatically.
        For local datasets, this is a no-op.
        """
        if self.use_local:
            # Local dataset - nothing to download
            return

        # Trigger download by loading a small portion
        # Use data_dir as cache directory if provided
        try:
            load_dataset(
                self.dataset_name,
                split=f"{self.split_mapping.get('train', 'train')}[:1]",
                trust_remote_code=self.trust_remote_code,
                cache_dir=str(self.data_dir) if self.data_dir is not None else None,
            )
        except Exception:
            pass  # Dataset may not have train split

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        Called once per process in distributed training.

        When caching is enabled, the cache contains PREPROCESSED data.
        """
        if stage == "fit" or stage is None:
            # === VALIDATION DATASET ===
            val_hf = self._load_hf_dataset("val")
            if val_hf is not None:
                if self.cache_dataset:
                    self.shared_val_images, self.shared_val_masks = self._build_shared_cache("val")
                    self.val_dataset = HuggingFaceSemanticDataset(
                        dataset_name=self.dataset_name,
                        split=self.split_mapping.get("val", "validation"),
                        image_column=self.image_column,
                        mask_column=self.mask_column,
                        transforms=self.val_transforms,
                        cache_dataset=self.cache_dataset,
                        cached_images=self.shared_val_images,
                        cached_masks=self.shared_val_masks,
                        id2label=self._id2label,
                        num_classes=self._num_classes,
                        hf_dataset=val_hf,
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    self.val_dataset = HuggingFaceSemanticDataset(
                        dataset_name=self.dataset_name,
                        split=self.split_mapping.get("val", "validation"),
                        image_column=self.image_column,
                        mask_column=self.mask_column,
                        transforms=self.val_transforms,
                        id2label=self._id2label,
                        num_classes=self._num_classes,
                        hf_dataset=val_hf,
                        trust_remote_code=self.trust_remote_code,
                    )

            # === TRAINING DATASET ===
            train_hf = self._load_hf_dataset("train")
            if train_hf is not None:
                if self.cache_dataset:
                    self.shared_train_images, self.shared_train_masks = self._build_shared_cache("train")
                    self.train_dataset = HuggingFaceSemanticDataset(
                        dataset_name=self.dataset_name,
                        split=self.split_mapping.get("train", "train"),
                        image_column=self.image_column,
                        mask_column=self.mask_column,
                        transforms=self.train_transforms,
                        cache_dataset=self.cache_dataset,
                        cached_images=self.shared_train_images,
                        cached_masks=self.shared_train_masks,
                        id2label=self._id2label,
                        num_classes=self._num_classes,
                        hf_dataset=train_hf,
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    self.train_dataset = HuggingFaceSemanticDataset(
                        dataset_name=self.dataset_name,
                        split=self.split_mapping.get("train", "train"),
                        image_column=self.image_column,
                        mask_column=self.mask_column,
                        transforms=self.train_transforms,
                        id2label=self._id2label,
                        num_classes=self._num_classes,
                        hf_dataset=train_hf,
                        trust_remote_code=self.trust_remote_code,
                    )

            # Store metadata
            if self.train_dataset is not None:
                self.num_classes = self.train_dataset.num_classes
                self.id2label = self.train_dataset.id2label

                # Validate consistency
                if self.val_dataset is not None:
                    if self.train_dataset.num_classes != self.val_dataset.num_classes:
                        print(
                            f"Warning: Train ({self.train_dataset.num_classes}) and "
                            f"val ({self.val_dataset.num_classes}) have different class counts"
                        )

        if stage == "test" or stage is None:
            # === TEST DATASET ===
            test_hf = self._load_hf_dataset("test")
            if test_hf is not None:
                if self.cache_dataset:
                    self.shared_test_images, self.shared_test_masks = self._build_shared_cache("test")
                    self.test_dataset = HuggingFaceSemanticDataset(
                        dataset_name=self.dataset_name,
                        split=self.split_mapping.get("test", "test"),
                        image_column=self.image_column,
                        mask_column=self.mask_column,
                        transforms=self.val_transforms,
                        cache_dataset=self.cache_dataset,
                        cached_images=self.shared_test_images,
                        cached_masks=self.shared_test_masks,
                        id2label=self._id2label,
                        num_classes=self._num_classes,
                        hf_dataset=test_hf,
                        trust_remote_code=self.trust_remote_code,
                    )
                else:
                    self.test_dataset = HuggingFaceSemanticDataset(
                        dataset_name=self.dataset_name,
                        split=self.split_mapping.get("test", "test"),
                        image_column=self.image_column,
                        mask_column=self.mask_column,
                        transforms=self.val_transforms,
                        id2label=self._id2label,
                        num_classes=self._num_classes,
                        hf_dataset=test_hf,
                        trust_remote_code=self.trust_remote_code,
                    )

    def train_dataloader(self) -> DataLoader:
        """Training dataloader with shuffling."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not available. Call setup('fit') first.")
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
        if self.val_dataset is None:
            raise RuntimeError("Val dataset not available. Call setup('fit') first.")
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
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not available. Call setup('test') first.")
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

    def collate_fn(self, batch):
        """
        Collate function to batch samples.

        When caching is enabled, data is ALREADY preprocessed.
        """
        images, masks = zip(*batch)

        # When using cache: just stack preprocessed tensors (uniform size)
        if self.cache_dataset:
            images = torch.stack(images, dim=0)
            masks = torch.stack(masks, dim=0)
            return {
                "pixel_values": images,
                "labels": masks,
            }

        # Apply preprocessing on-the-fly if preprocessor is provided
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

    def get_num_classes(self) -> int:
        """Get number of classes (including background)."""
        if self.num_classes is None:
            raise RuntimeError("Call setup() or fit() before accessing num_classes")
        return self.num_classes

    def get_id2label(self) -> Dict[int, str]:
        """Get mapping from class IDs to labels."""
        if self.id2label is None:
            raise RuntimeError("Call setup() or fit() before accessing id2label")
        return self.id2label

    def teardown(self, stage: Optional[str] = None):
        """
        Clean up shared memory resources.
        """
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

    def __repr__(self) -> str:
        """String representation with key info."""
        if self.train_dataset is not None:
            cache_status = "preprocessed & cached" if self.cache_dataset else "not cached"
            data_source = str(self.data_dir) if self.use_local else self.dataset_name
            return (
                f"{self.__class__.__name__}(\n"
                f"  data_source={data_source},\n"
                f"  local={self.use_local},\n"
                f"  num_classes={self.num_classes},\n"
                f"  img_size={self.img_size},\n"
                f"  preprocessor={'HuggingFace' if self.preprocessor else 'PyTorch resize'},\n"
                f"  train_size={len(self.train_dataset)} ({cache_status}),\n"
                f"  val_size={len(self.val_dataset) if self.val_dataset else 0} ({cache_status}),\n"
                f"  test_size={len(self.test_dataset) if self.test_dataset else 0} ({cache_status}),\n"
                f"  batch_size={self.batch_size},\n"
                f"  num_workers={self.num_workers}\n"
                f")"
            )

        return f"{self.__class__.__name__}(not setup yet)"

    def __len__(self) -> int:
        """Return number of training batches."""
        if self.train_dataset is None:
            raise RuntimeError("Call setup() or fit() before accessing length")
        return len(self.train_dataset) // self.batch_size
