from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from typing import Optional, Dict, Any
import numpy as np


class HuggingFaceSemanticDataset(Dataset):
    """
    HuggingFace dataset wrapper for semantic segmentation.

    Label contract (STRICT):
        0  -> background (if fill_background=True)
        1  -> first category
        ...
        N  -> Nth category

    Returns:
        image: FloatTensor [3, H, W] in [0, 1]
        mask:  LongTensor  [H, W] with values in [0, N]
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        image_column: str = "image",
        mask_column: str = "label",
        transforms: T.Compose | None = None,
        cache_dataset: bool = False,
        fill_background: bool = True,
        id2label: Optional[Dict[int, str]] = None,
        num_classes: Optional[int] = None,
        cached_images: Optional[torch.Tensor] = None,
        cached_masks: Optional[torch.Tensor] = None,
        hf_dataset: Optional[Any] = None,  # Pre-loaded dataset
        trust_remote_code: bool = False,
    ):
        """
        Args:
            dataset_name: HuggingFace dataset name (e.g., "scene_parse_150")
            split: Dataset split ("train", "validation", "test")
            image_column: Column name for images in the dataset
            mask_column: Column name for segmentation masks
            transforms: Torchvision transforms to apply
            cache_dataset: Whether to use shared memory cache
            fill_background: Whether to reserve index 0 for background
            id2label: Optional mapping from class IDs to labels
            num_classes: Optional number of classes (inferred if not provided)
            cached_images: Pre-built shared memory image cache
            cached_masks: Pre-built shared memory mask cache
            hf_dataset: Pre-loaded HuggingFace dataset (avoids reloading)
            trust_remote_code: Whether to trust remote code for dataset loading
        """
        self.dataset_name = dataset_name
        self.split = split
        self.image_column = image_column
        self.mask_column = mask_column
        self.transforms = transforms
        self.cache_dataset = cache_dataset
        self.fill_background = fill_background
        self.trust_remote_code = trust_remote_code

        # Load dataset from HuggingFace or use pre-loaded
        if hf_dataset is not None:
            self.dataset = hf_dataset
        else:
            self.dataset = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=trust_remote_code,
            )

        # Set up class mappings
        if id2label is not None:
            self.id2label = id2label
            self.num_classes = len(id2label)
        elif num_classes is not None:
            self.num_classes = num_classes
            self.id2label = {i: f"class_{i}" for i in range(num_classes)}
        else:
            # Try to infer from dataset features
            self._infer_class_info()

        # Optional cache
        self._image_cache = cached_images
        self._mask_cache = cached_masks

    def _infer_class_info(self):
        """Infer class information from HuggingFace dataset features."""
        try:
            # Try to get from dataset features
            features = self.dataset.features
            if self.mask_column in features:
                mask_feature = features[self.mask_column]
                # Check if it has names attribute (ClassLabel)
                if hasattr(mask_feature, "names"):
                    names = mask_feature.names
                    self.num_classes = len(names)
                    self.id2label = {i: name for i, name in enumerate(names)}
                    return
                # Check for feature metadata
                if hasattr(mask_feature, "feature") and hasattr(mask_feature.feature, "names"):
                    names = mask_feature.feature.names
                    self.num_classes = len(names)
                    self.id2label = {i: name for i, name in enumerate(names)}
                    return
        except Exception:
            pass

        # Fallback: scan first few samples to find max class
        print("Inferring num_classes from data samples...")
        max_class = 0
        for i in range(min(100, len(self.dataset))):
            sample = self.dataset[i]
            mask = sample[self.mask_column]
            if isinstance(mask, Image.Image):
                mask = np.array(mask)
            max_class = max(max_class, mask.max())

        self.num_classes = int(max_class) + 1
        self.id2label = {i: f"class_{i}" for i in range(self.num_classes)}
        print(f"Inferred num_classes={self.num_classes}")

    def _load_image(self, index: int) -> tv_tensors.Image:
        """Load and convert image to tensor."""
        sample = self.dataset[index]
        image = sample[self.image_column]

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        return tv_tensors.Image(image)

    def _load_mask(self, index: int) -> tv_tensors.Mask:
        """Load and convert mask to tensor."""
        sample = self.dataset[index]
        mask = sample[self.mask_column]

        if isinstance(mask, Image.Image):
            mask = torch.from_numpy(np.array(mask)).long()
        elif isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        elif isinstance(mask, torch.Tensor):
            mask = mask.long()

        return tv_tensors.Mask(mask)

    def _load_sample(self, index: int):
        """Load a sample from the dataset."""
        image = self._load_image(index)
        mask = self._load_mask(index)
        return image, mask

    def __getitem__(self, index: int):
        """Get item from shared cache or load from HuggingFace."""
        if self.cache_dataset and self._image_cache is not None:
            # Load from shared memory
            image = self._image_cache[index].clone()
            mask = self._mask_cache[index].clone()
        else:
            # Load from HuggingFace dataset
            image, mask = self._load_sample(index)

        # Apply transforms (e.g., augmentation for training)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)

        return image, mask

    def __len__(self) -> int:
        return len(self.dataset)
