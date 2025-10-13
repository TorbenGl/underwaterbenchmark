import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
from PIL import Image

class RandomResizeCropSeg:
    def __init__(self,  scale=(0.5, 1.2), ratio=(0.75, 1.33), hflip_prob=0.5):        
        self.scale = scale
        self.ratio = ratio
        self.hflip_prob = hflip_prob

    def __call__(self, img, mask):
        # --- Random resized crop ---
        size = (img.shape[-2], img.shape[-1])
        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, size, interpolation=Image.BILINEAR)
        mask = F.resized_crop(mask, i, j, h, w, size, interpolation=Image.NEAREST)
        # --- Random horizontal flip (applied equally to both) ---
        if random.random() < self.hflip_prob:
            img = F.hflip(img)
            mask = F.hflip(mask)
        return img, mask