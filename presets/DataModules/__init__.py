"""
Dataset presets for semantic segmentation training.

Usage:
    from presets.DataModules import get_datamodule, AVAILABLE_DATASETS, set_data_root

    # Set global data root (all datasets stored under this path)
    set_data_root("/workspace/data")

    # List available datasets
    print(AVAILABLE_DATASETS)

    # Get a dataset by name
    datamodule = get_datamodule(
        name="cou",
        batch_size=8,
        num_workers=4,
        preprocessor=preprocessor,
    )
"""

import os
from typing import Optional, Dict, Type, Union
from datamodules.cocodatamodule_semantic import CocoLightningDataModule_Semantic
from datamodules.imagemask_datamodule import ImageMaskDataModule
from datamodules.hf_datamodule_semantic import HuggingFaceLightningDataModule_Semantic


# =============================================================================
# GLOBAL DATA ROOT CONFIGURATION
# =============================================================================

# Default data root - can be overridden via set_data_root() or DATA_ROOT env var
_DATA_ROOT: str = os.environ.get("DATA_ROOT", "/workspace/data")


def set_data_root(path: str) -> None:
    """
    Set the global data root directory.

    All dataset presets will use this as their base path.
    Individual datasets specify their subdirectory relative to this root.

    Args:
        path: Absolute path to the data root directory

    Example:
        set_data_root("/workspace/data")
        # COU dataset will look in: /workspace/data/cou/coco/coco/
    """
    global _DATA_ROOT
    _DATA_ROOT = path
    print(f"Data root set to: {_DATA_ROOT}")


def get_data_root() -> str:
    """Get the current global data root directory."""
    return _DATA_ROOT


class COCODatamodulePreset(CocoLightningDataModule_Semantic):
    """
    Base class for dataset presets.

    Subclasses should define:
        - name: Dataset identifier
        - data_subdir: Subdirectory relative to global DATA_ROOT (e.g., "cou/coco/coco")
        - image_folder: Folder containing images (relative to data_subdir)
        - annotation_files: Dict with train/val/test annotation paths
        - default_img_size: Default image size (H, W)
        - ignore_index: Label index to ignore
        - increase_idx: Whether to increase class indices by 1
        - fill_background: Whether to fill unlabeled pixels as background

    The full data path is constructed as: DATA_ROOT / data_subdir
    Example: /workspace/data/cou/coco/coco/
    """

    # Override these in subclasses
    name: str = "base"
    data_subdir: str = ""  # Subdirectory relative to DATA_ROOT
    image_folder: Union[str, Dict[str, str]] = "images"  # Can be str or {"train": ..., "val": ..., "test": ...}
    annotation_files: Dict[str, str] = {}
    default_img_size: tuple = (512, 512)
    loss_ignore_index: Optional[int] = 255
    metric_ignore_index: Optional[int] = 255
    increase_idx: bool = False
    fill_background: bool = False
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None

    @classmethod
    def get_metadata(cls, img_size: Optional[tuple] = None) -> Dict:
        """
        Get dataset metadata without instantiating or calling setup().

        Args:
            img_size: Override image size. If None, uses default_img_size.

        Returns:
            Dict with keys: num_classes, img_size, ignore_index, id2label, label2id
        """
        num_classes = len(cls.id2label) if cls.id2label else None
        return {
            "num_classes": num_classes,
            "img_size": img_size or cls.default_img_size,
            "loss_ignore_index": cls.loss_ignore_index if cls.loss_ignore_index is not None else 255,
            "metric_ignore_index": cls.metric_ignore_index if cls.metric_ignore_index is not None else 255,
            "id2label": cls.id2label or {},
            "label2id": cls.label2id or {},
        }

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        img_size: Optional[tuple] = None,
        preprocessor=None,
        root: Optional[str] = None,
        cache_dataset: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        devices: str = "auto",
    ):
        """
        Initialize the dataset preset.

        Args:
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            img_size: Image size (H, W). Uses default if None.
            preprocessor: preprocessor (optional)
            root: Full path override. If None, uses DATA_ROOT + data_subdir.
            cache_dataset: Whether to cache dataset in shared memory
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Batches to prefetch per worker
            devices: GPU devices specification
        """
        # Construct full path: use override or DATA_ROOT + data_subdir
        if root is not None:
            data_path = root
        else:
            data_path = os.path.join(get_data_root(), self.data_subdir)

        super().__init__(
            root=data_path,
            image_folder=self.image_folder,
            annotation_file_dict=self.annotation_files,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size or self.default_img_size,
            preprocessor=preprocessor,
            devices=devices,
            ignore_idx=self.ignore_index,
            increase_idx=self.increase_idx,
            fill_background=self.fill_background,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            cache_dataset=cache_dataset,
            prefetch_factor=prefetch_factor,
            train_transforms=None,  # Augmentations handled in LightningModule
            val_transforms=None,
            id2label=self.id2label,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# IMAGE-MASK DATASET PRESET (for simple image/mask pair datasets)
# =============================================================================

class ImageMaskDatamodulePreset(ImageMaskDataModule):
    """
    Base class for image-mask dataset presets.

    Use this for datasets with simple image/mask pairs (not COCO format).

    Subclasses should define:
        - name: Dataset identifier
        - data_subdir: Subdirectory relative to global DATA_ROOT
        - train_annotation_file: Path to train annotations (relative to data_subdir)
        - val_annotation_file: Path to val annotations (relative to data_subdir)
        - test_annotation_file: Path to test annotations (optional)
        - default_img_size: Default image size (H, W)
        - rgb_mask: Whether masks are RGB color-coded
        - background_color: RGB color for background class
        - color_to_class: Optional explicit color-to-class mapping

    Annotation file format (JSON):
        {
            "samples": [
                {"image": "path/to/img.jpg", "mask": "path/to/mask.png"},
                ...
            ]
        }
    """

    # Override these in subclasses
    name: str = "base_imagemask"
    data_subdir: str = ""
    train_annotation_file: Optional[str] = None
    val_annotation_file: Optional[str] = None
    test_annotation_file: Optional[str] = None
    default_img_size: tuple = (512, 512)
    num_classes: Optional[int] = None
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    rgb_mask: bool = False
    color_to_class: Optional[Dict[tuple, int]] = None
    background_color: tuple = (0, 0, 0)
    increase_idx: bool = False  # Normalize binary masks (0/255 -> 0/1)
    value_to_class: Optional[Dict[int, int]] = None  # Map specific pixel values to class indices
    loss_ignore_index: Optional[int] = 255
    metric_ignore_index: Optional[int] = 255
    @classmethod
    def get_metadata(cls, img_size: Optional[tuple] = None) -> Dict:
        """
        Get dataset metadata without instantiating or calling setup().

        Args:
            img_size: Override image size. If None, uses default_img_size.

        Returns:
            Dict with keys: num_classes, img_size, ignore_index, id2label, label2id
        """
        num_classes = len(cls.id2label) if cls.id2label else None
        return {
            "num_classes": num_classes,
            "img_size": img_size or cls.default_img_size,
            "loss_ignore_index": cls.loss_ignore_index if cls.loss_ignore_index is not None else 255,
            "metric_ignore_index": cls.metric_ignore_index if cls.metric_ignore_index is not None else 255,
            "id2label": cls.id2label or {},
            "label2id": cls.label2id or {},
        }
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        img_size: Optional[tuple] = None,
        preprocessor=None,
        root: Optional[str] = None,
        cache_dataset: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        devices: str = "auto",
    ):
        """
        Initialize the image-mask dataset preset.

        Args:
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            img_size: Image size (H, W). Uses default if None.
            preprocessor: HuggingFace preprocessor (optional)
            root: Full path override. If None, uses DATA_ROOT + data_subdir.
            cache_dataset: Whether to cache dataset in shared memory
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Batches to prefetch per worker
            devices: GPU devices specification
        """
        # Construct full path: use override or DATA_ROOT + data_subdir
        if root is not None:
            data_path = root
        else:
            data_path = os.path.join(get_data_root(), self.data_subdir)

        super().__init__(
            root=data_path,
            train_annotation_file=self.train_annotation_file,
            val_annotation_file=self.val_annotation_file,
            test_annotation_file=self.test_annotation_file,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size or self.default_img_size,
            preprocessor=preprocessor,
            num_classes=self.num_classes,
            id2label=self.id2label,
            rgb_mask=self.rgb_mask,
            color_to_class=self.color_to_class,
            background_color=self.background_color,
            increase_idx=self.increase_idx,
            devices=devices,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            cache_dataset=cache_dataset,
            prefetch_factor=prefetch_factor,
            value_to_class=self.value_to_class,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# HUGGINGFACE DATASET PRESET
# =============================================================================

class HuggingFaceDatamodulePreset(HuggingFaceLightningDataModule_Semantic):
    """
    Base class for HuggingFace dataset presets.

    Supports both:
    - Loading from HuggingFace Hub (hf_dataset_name="EPFL-ECEO/coralscapes")
    - Loading from local directory (data_subdir="coralscapes")

    Subclasses should define:
        - name: Dataset identifier
        - hf_dataset_name: Full HuggingFace dataset name (for Hub loading)
        - data_subdir: Subdirectory relative to DATA_ROOT (for local loading)
        - image_column: Column name for images (default: "image")
        - mask_column: Column name for masks (default: "label")
        - split_mapping: Dict mapping standard splits to HuggingFace split names
        - default_img_size: Default image size (H, W)
        - num_classes: Number of classes (optional, auto-inferred if not set)
        - id2label: Optional dict mapping class indices to label names
        - trust_remote_code: Whether to trust remote code for dataset loading

    The full local path is constructed as: DATA_ROOT / data_subdir
    Example: /workspace/data/coralscapes/
    """

    # Override these in subclasses
    name: str = "base_hf"
    hf_dataset_name: Optional[str] = None  # Full HuggingFace dataset name (for Hub)
    data_subdir: Optional[str] = None  # Subdirectory relative to DATA_ROOT (for local)
    image_column: str = "image"
    mask_column: str = "label"
    split_mapping: Optional[Dict[str, str]] = None
    default_img_size: tuple = (512, 512)
    num_classes: Optional[int] = None
    id2label: Optional[Dict[int, str]] = None
    label2id: Optional[Dict[str, int]] = None
    trust_remote_code: bool = False
    loss_ignore_index: Optional[int] = 255
    metric_ignore_index: Optional[int] = 255

    @classmethod
    def get_metadata(cls, img_size: Optional[tuple] = None) -> Dict:
        """
        Get dataset metadata without instantiating or calling setup().

        Args:
            img_size: Override image size. If None, uses default_img_size.

        Returns:
            Dict with keys: num_classes, img_size, ignore_index, id2label, label2id
        """
        num_classes = len(cls.id2label) if cls.id2label else None
        return {
            "num_classes": num_classes,
            "img_size": img_size or cls.default_img_size,
            "loss_ignore_index": cls.loss_ignore_index if cls.loss_ignore_index is not None else 255,
            "metric_ignore_index": cls.metric_ignore_index if cls.metric_ignore_index is not None else 255,
            "id2label": cls.id2label or {},
            "label2id": cls.label2id or {},
        }

    def __init__(
        self,
        batch_size: int,
        num_workers: int = 4,
        img_size: Optional[tuple] = None,
        preprocessor=None,
        root: Optional[str] = None,
        cache_dataset: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        prefetch_factor: int = 2,
        devices: str = "auto",
    ):
        """
        Initialize the HuggingFace dataset preset.

        Args:
            batch_size: Batch size for training
            num_workers: Number of data loading workers
            img_size: Image size (H, W). Uses default if None.
            preprocessor: HuggingFace preprocessor (optional)
            root: Full path override for local loading. If None, uses DATA_ROOT + data_subdir.
            cache_dataset: Whether to cache dataset in shared memory
            pin_memory: Pin memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
            prefetch_factor: Batches to prefetch per worker
            devices: GPU devices specification
        """
        # Determine data source: local path or HuggingFace Hub
        data_dir = None
        dataset_name = None

        if root is not None:
            # Explicit root path override
            data_dir = root
        elif self.data_subdir is not None:
            # Local loading using DATA_ROOT + data_subdir
            data_dir = os.path.join(get_data_root(), self.data_subdir)
        elif self.hf_dataset_name is not None:
            # HuggingFace Hub loading
            dataset_name = self.hf_dataset_name
        else:
            raise ValueError(
                f"Dataset '{self.name}' must define either hf_dataset_name or data_subdir"
            )

        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size or self.default_img_size,
            dataset_name=dataset_name,
            data_dir=data_dir,
            split_mapping=self.split_mapping,
            image_column=self.image_column,
            mask_column=self.mask_column,
            preprocessor=preprocessor,
            devices=devices,
            id2label=self.id2label,
            num_classes=self.num_classes,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            cache_dataset=cache_dataset,
            prefetch_factor=prefetch_factor,
            trust_remote_code=self.trust_remote_code,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# DATASET REGISTRY
# =============================================================================

# Registry of available dataset presets (supports all types)
_DATASET_REGISTRY: Dict[str, Type[Union[COCODatamodulePreset, ImageMaskDatamodulePreset, HuggingFaceDatamodulePreset]]] = {}


def register_dataset(cls: Type[Union[COCODatamodulePreset, ImageMaskDatamodulePreset, HuggingFaceDatamodulePreset]]) -> Type[Union[COCODatamodulePreset, ImageMaskDatamodulePreset, HuggingFaceDatamodulePreset]]:
    """Decorator to register a dataset preset (works with both COCO and ImageMask types)."""
    _DATASET_REGISTRY[cls.name] = cls
    return cls


def get_datamodule(name: str, **kwargs) -> Union[COCODatamodulePreset, ImageMaskDatamodulePreset, HuggingFaceDatamodulePreset]:
    """
    Get a dataset preset by name.

    Args:
        name: Dataset name (e.g., "cou", "suim", "trash")
        **kwargs: Arguments passed to the dataset constructor

    Returns:
        Initialized DatasetPreset instance

    Raises:
        ValueError: If dataset name is not found
    """
    if name not in _DATASET_REGISTRY:
        available = ", ".join(_DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unknown dataset: '{name}'. Available datasets: {available}"
        )
    return _DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> list:
    """List all available dataset names."""
    return list(_DATASET_REGISTRY.keys())


# Import preset modules to trigger registration
from presets.DataModules.cou import COUDataset
from presets.DataModules.liaci import LIACiDataset
from presets.DataModules.l4s import L4SDataset
from presets.DataModules.coralscapes import CoralscapesDataset
from presets.DataModules.suim import SUIMDataset
from presets.DataModules.trashcan import (
    TrashCanInstanceDataset,
    TrashCanMaterialDataset,
)
from presets.DataModules.seaclear import (
    SeaClearBaseDataset,
    SeaClearMaterialDataset,
    SeaClearSuperclassDataset,
)
from presets.DataModules.uiis10k import UIIS10KDataset
from presets.DataModules.usod10k import USOD10KDataset
from presets.DataModules.coralmask import CoralMaskDataset

# For convenience
AVAILABLE_DATASETS = list_datasets()
