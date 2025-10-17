
import torch
from torchvision import transforms
import numpy as np

class DinoPreprocessor:
    def __init__(self, config, ignore_index=-1):
        self.config = config
        self.ignore_index = ignore_index
        self.transform = transforms.Compose([                 
                                        transforms.Normalize(mean=self.config['image_mean'],  
                                        std=self.config['image_mean'])                                          
                                    ])
        self.sizedivisior = self.config.get("size_divisor", None)

    def __call__(self, images, segmentation_maps=None, size=None, return_tensors="pt", ignore_index=-1):
        output = {}        
        # Apply transformations to each image in the batch#
        size 
        images = [self.transform(torch.tensor(img).to(torch.get_default_dtype())) if isinstance(img, np.ndarray) else img for img in images]      
        resized_images = transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC)(torch.stack(images))
        segmentation_maps = [torch.tensor(mp).to(torch.get_default_dtype()) if isinstance(mp, np.ndarray) else mp for mp in segmentation_maps] if segmentation_maps is not None else None
        segmentation_maps = transforms.Resize(size, interpolation=transforms.InterpolationMode.NEAREST)(torch.stack(segmentation_maps)) if segmentation_maps is not None else None
        segmentation_maps = segmentation_maps.long() if segmentation_maps is not None else None
        output["pixel_values"] = resized_images
        output["mask_labels"] = segmentation_maps

        if return_tensors == "np":
            output = {k: v.cpu().numpy() for k, v in output.items()}
        

        return output



        
        
