from pycocotools.coco import COCO
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from typing import Optional


class COCOSemanticDataset(Dataset):
    """
    Robust COCO-style dataset for semantic segmentation.

    Label contract (STRICT):
        0  -> background (synthetic)
        1  -> first category
        ...
        N  -> Nth category

    Returns:
        image: FloatTensor [3, H, W] in [0, 1]
        mask:  LongTensor  [H, W] with values in [0, N]
    """

    def __init__(
        self,
        root: str,
        annotation_file: str,
        image_folder: str,
        transforms: T.Compose | None = None,
        cache_dataset: bool = False,
        fill_background: bool = True,
        increase_idx: bool = False,
        cached_images: Optional[torch.Tensor] = None,
        cached_masks: Optional[torch.Tensor] = None,
    ):
        self.root = Path(root)
        self.image_folder = self.root / image_folder
        self.transforms = transforms
        self.cache_dataset = cache_dataset
        self.fill_background = fill_background

        # ----------------------------------------------------
        # Load COCO
        # ----------------------------------------------------
        self.coco = COCO(self.root / annotation_file)
        self.ids = sorted(self.coco.imgs.keys())

        # ----------------------------------------------------
        # Category remapping (ALWAYS background = 0)
        # ----------------------------------------------------
        self.background_id = 0
        self.increase_idx = increase_idx
        cat_ids = sorted(self.coco.getCatIds())

        if self.fill_background:
            self.cat_id_map = {
            cat_id: idx + 1
            for idx, cat_id in enumerate(cat_ids)
            }
            self.id2label = {
                0: "background",
                **{
                    idx + int(self.increase_idx): self.coco.loadCats(cat_id)[0]["name"]
                    for idx, cat_id in enumerate(cat_ids)
                }
            }
        else:
            self.cat_id_map = {
                cat_id: idx
                for idx, cat_id in enumerate(cat_ids)
            }
            self.id2label = {
                **{
                    idx: self.coco.loadCats(cat_id)[0]["name"]
                    for idx, cat_id in enumerate(cat_ids)
                }
            }

        self.num_classes = len(self.cat_id_map) + 1  # + background

        # Optional cache
        self._image_cache = cached_images
        self._mask_cache = cached_masks
    

    def _load_image(self, img_id):
        info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_folder / info["file_name"]
        image = Image.open(img_path).convert("RGB")
        return tv_tensors.Image(image)

    def _load_mask(self, img_id, height, width):
        # Background initialized to 0
        mask = torch.zeros((height, width), dtype=torch.long)
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        for ann in anns:
            mapped_id = self.cat_id_map[ann["category_id"]]
            binary_mask = self.coco.annToMask(ann)
            # Handle dimension mismatch (e.g., EXIF rotation in PIL vs annotation dims)
            if binary_mask.shape != (height, width):
                if binary_mask.shape == (width, height):
                    # Dimensions are swapped - transpose
                    binary_mask = binary_mask.T
                else:
                    # Different dimensions entirely - resize
                    import cv2
                    binary_mask = cv2.resize(
                        binary_mask.astype('uint8'),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )
            mask[binary_mask == 1] = mapped_id
        return tv_tensors.Mask(mask)    

    def _load_sample(self, index):
        img_id = self.ids[index]
        image = self._load_image(img_id)
        _, h, w = image.shape
        mask = self._load_mask(img_id, h, w)
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)        
        return image, mask
    

    def __getitem__(self, index):
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

    def __len__(self):
        return len(self.ids)