import lightning
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from torch.utils.data import DataLoader
import torch
import torch.multiprocessing as mp
import torchvision.transforms as T
from pathlib import Path
from typing import Optional, Tuple, Dict, Union
from segdatasets.cocodataset_semantic import COCOSemanticDataset


class CocoLightningDataModule_Semantic(lightning.LightningDataModule):
    """
    Lightning DataModule for COCO Semantic Segmentation.

    Key Features:
    - Preprocessor is applied BEFORE caching (cache stores preprocessed data)
    - Shared memory caching for efficient distributed training
    - All workers access the same cached tensors without duplication
    - Separate control for caching train/val/test splits
    - Supports per-split image folders via dict (e.g., {"train": "train", "val": "val"})

    WARNING: Caching requires significant RAM. For COCO at 512x512,
    expect ~50GB+ of memory usage. Ensure sufficient system resources.
    """

    def __init__(
        self,
        root: str,
        image_folder: Union[str, Dict[str, str]],
        annotation_file_dict: dict[str, str],
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        preprocessor=None,  # Applied BEFORE caching if provided
        devices: str | list = "auto",
        ignore_idx: Optional[int] = None,
        increase_idx: bool = False,
        fill_background: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        cache_dataset: bool = False,  # Master switch for all caching
        prefetch_factor: int = 2,
        train_transforms=None,  # Applied AFTER cache retrieval (augmentations only)
        val_transforms=None,    # Applied AFTER cache retrieval
        id2label: Optional[Dict[int, str]] = None,
    ) -> None:
        super().__init__()
        
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
        self.root = Path(root)
        self.image_folder = image_folder
        self.annotation_file_dict = annotation_file_dict
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.preprocessor = preprocessor
        self.ignore_idx = ignore_idx
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.increase_idx = increase_idx
        self.fill_background = fill_background
        
        # Caching configuration with per-split control
        self.cache_dataset = cache_dataset        
        
        # Parse devices (kept for potential future use)
        self.devices = (
            devices if devices == "auto" 
            else _parse_gpu_ids(devices, include_cuda=True)
        )
        
        # Transforms (applied AFTER cache retrieval - should be augmentations only)
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

        # Class info (id2label can be provided or extracted from dataset after setup)
        self._id2label = id2label
        self.id2label = None

        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_classes = None
        
        # Shared memory tensors (created in setup())
        self.shared_val_images = None
        self.shared_val_masks = None
        self.shared_train_images = None
        self.shared_train_masks = None
        self.shared_test_images = None
        self.shared_test_masks = None

    def _get_image_folder(self, split: str) -> str:
        """Get image folder for a specific split.

        Args:
            split: One of "train", "val", or "test"

        Returns:
            Image folder path for the given split
        """
        if isinstance(self.image_folder, dict):
            return self.image_folder[split]
        return self.image_folder

    def _build_shared_cache(
        self,
        annotation_file: str,
        split_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build shared memory cache with PREPROCESSED data.
        Process flow:
        1. Load raw image from disk
        2. Apply preprocessor/resize
        3. Store preprocessed result in shared memory
        This means the cache contains ready-to-use tensors. During training,
        workers retrieve preprocessed data and only apply augmentations
        (specified in train_transforms/val_transforms).
        IMPORTANT: In distributed training, each process calls this independently.
        For truly efficient shared memory, consider having only rank 0 build
        the cache and synchronize across processes.
        Returns:
            shared_images: Tensor [N, C, H, W] in shared memory (preprocessed)
            shared_masks: Tensor [N, H, W] in shared memory (preprocessed)
        """
        print(f"Building shared memory cache for {split_name} split...")
        print(f"  Preprocessing will be applied before caching")
        if self.preprocessor is not None:
            print(f"  Using  preprocessor")
        else:
            print(f"  Using standard resize to {self.img_size}")

        # Get split-specific image folder
        image_folder = self._get_image_folder(split_name)

        # Create temporary dataset for loading
        temp_dataset = COCOSemanticDataset(
            root=self.root,
            annotation_file=annotation_file,
            image_folder=image_folder,
            increase_idx=self.increase_idx,
            fill_background=self.fill_background,
            transforms=None,  # No transforms during caching
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
            # Load raw sample from disk
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
                
        print(f"Caching completed for {split_name}. Data is preprocessed and ready.")
        return shared_images, shared_masks

    def prepare_data(self):
        """
        Download/prepare data if needed. Called only on rank 0 in distributed training.
        For COCO, we assume data is already downloaded, so this is a no-op.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """
        Setup datasets for each stage.
        Called once per process in distributed training.        
        When caching is enabled, the cache contains PREPROCESSED data.
        The dataset will retrieve preprocessed tensors and only apply
        additional transforms (augmentations) specified in train_transforms/val_transforms.
        """
        if stage == "fit" or stage is None:
            # === VALIDATION DATASET ===
            if self.cache_dataset:
                # Build cache with preprocessed data
                self.shared_val_images, self.shared_val_masks = self._build_shared_cache(
                    self.annotation_file_dict["val"],
                    "val"
                )
                self.val_dataset = COCOSemanticDataset(
                    root=self.root,
                    annotation_file=self.annotation_file_dict["val"],
                    image_folder=self._get_image_folder("val"),
                    increase_idx=self.increase_idx,
                    cache_dataset=self.cache_dataset,
                    transforms=self.val_transforms,  # Only augmentations applied
                    cached_images=self.shared_val_images,
                    cached_masks=self.shared_val_masks,
                    fill_background=self.fill_background,
                )
                self.shared_train_images, self.shared_train_masks = self._build_shared_cache(
                    self.annotation_file_dict["train"],
                    "train"
                )
                self.train_dataset = COCOSemanticDataset(
                    root=self.root,
                    annotation_file=self.annotation_file_dict["train"],
                    image_folder=self._get_image_folder("train"),
                    transforms=self.train_transforms,  # Only augmentations applied
                    increase_idx=self.increase_idx,
                    cache_dataset=self.cache_dataset,
                    cached_images=self.shared_train_images,
                    cached_masks=self.shared_train_masks,
                    fill_background=self.fill_background,
                )
            else:
                # No caching: dataset handles all loading and preprocessing
                self.val_dataset = COCOSemanticDataset(
                    root=self.root,
                    annotation_file=self.annotation_file_dict["val"],
                    image_folder=self._get_image_folder("val"),
                    increase_idx=self.increase_idx,
                    transforms=self.val_transforms,
                    fill_background=self.fill_background,
                )
                # No caching: dataset handles all loading and preprocessing
                self.train_dataset = COCOSemanticDataset(
                    root=self.root,
                    annotation_file=self.annotation_file_dict["train"],
                    image_folder=self._get_image_folder("train"),
                    increase_idx=self.increase_idx,
                    transforms=self.train_transforms,
                    fill_background=self.fill_background,
                )            
            # Store metadata
            self.num_classes = self.train_dataset.num_classes
            # Use provided id2label or extract from dataset
            if self._id2label is not None:
                self.id2label = self._id2label
            else:
                self.id2label = self.train_dataset.id2label

            # Validate consistency
            assert self.train_dataset.num_classes == self.val_dataset.num_classes, \
                "Train/val splits have different number of classes!"
        
        if stage == "test" or stage is None:
            # === TEST DATASET ===
            if self.cache_dataset:
                # Build cache with preprocessed data
                self.shared_test_images, self.shared_test_masks = self._build_shared_cache(
                    self.annotation_file_dict["test"],
                    "test"
                )
                self.test_dataset = COCOSemanticDataset(
                    root=self.root,
                    annotation_file=self.annotation_file_dict["test"],
                    image_folder=self._get_image_folder("test"),
                    transforms=self.val_transforms,  # Use val transforms for test
                    increase_idx=self.increase_idx,
                    cache_dataset=self.cache_dataset,
                    cached_images=self.shared_test_images,
                    cached_masks=self.shared_test_masks,
                    fill_background=self.fill_background,
                )
            else:
                # No caching: dataset handles all loading and preprocessing
                self.test_dataset = COCOSemanticDataset(
                    root=self.root,
                    annotation_file=self.annotation_file_dict["test"],
                    image_folder=self._get_image_folder("test"),
                    transforms=self.val_transforms,
                    increase_idx=self.increase_idx,
                    fill_background=self.fill_background,
                )            
            
    
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
    
    def collate_fn(self, batch):
        """
        Collate function to batch samples.

        IMPORTANT: When caching is enabled, data is ALREADY preprocessed.
        The preprocessor is NOT applied here - we just stack tensors.

        This is much more efficient than applying preprocessing during
        training, as it happens once at cache-build time, not every batch.

        For variable-sized images (no cache), each sample is processed
        individually before stacking to handle different dimensions.
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
        # (handles variable-sized images)
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
        Properly releases all shared memory tensors to avoid memory leaks.
        """
        if stage == "fit":
            # Clean up validation cache
            if self.shared_val_images is not None:
                del self.shared_val_images
                del self.shared_val_masks
                self.shared_val_images = None
                self.shared_val_masks = None
            
            # Clean up training cache
            if self.shared_train_images is not None:
                del self.shared_train_images
                del self.shared_train_masks
                self.shared_train_images = None
                self.shared_train_masks = None        
        if stage == "test":
            # Clean up test cache
            if self.shared_test_images is not None:
                del self.shared_test_images
                del self.shared_test_masks
                self.shared_test_images = None
                self.shared_test_masks = None
    
    def __repr__(self) -> str:
        """String representation with key info."""
        if self.train_dataset is not None:
            train_cache = "preprocessed & cached" if self.cache_train else "not cached"
            val_cache = "preprocessed & cached" if self.cache_val else "not cached"
            test_cache = "preprocessed & cached" if self.cache_test else "not cached"            
            return (
                f"{self.__class__.__name__}(\n"
                f"  num_classes={self.num_classes},\n"
                f"  img_size={self.img_size},\n"
                f"  preprocessor={'HuggingFace' if self.preprocessor else 'PyTorch resize'},\n"
                f"  train_size={len(self.train_dataset)} ({train_cache}),\n"
                f"  val_size={len(self.val_dataset) if self.val_dataset else 0} ({val_cache}),\n"
                f"  test_size={len(self.test_dataset) if self.test_dataset else 0} ({test_cache}),\n"
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