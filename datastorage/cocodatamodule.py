from typing import Optional
import torch
import lightning
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from torch.utils.data import DataLoader 
from datastorage.cocodataset import COCODataset
import torchvision.transforms.v2 as transforms
from datastorage.transforms import RandomResizeCropSeg

class CocoLightningDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        path,
        image_folder,
        annotation_file_dict,
        fill_background,
        devices,
        batch_size: int,
        num_workers: int,
        img_size: tuple[int, int],
        id2label: dict[int, str],
        preprocessor,
        ignore_idx: Optional[int] = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
                ) -> None:
        super().__init__()        
        self.path = path
        self.ignore_idx = ignore_idx # Introduce later
        self.img_size = img_size
        self.image_folder = image_folder
        self.annotation_file_dict = annotation_file_dict
        self.fill_background = fill_background
        self.id2label = id2label
        self.preprocessor = preprocessor

        self.devices = (
            devices if devices == "auto" else _parse_gpu_ids(devices, include_cuda=True)
        )
        
        self.batch_size= batch_size

        self.num_workers = num_workers

        self.train_transforms = RandomResizeCropSeg(size=(1080, 1920))
        self.val_transforms = None
        self.test_transforms = None

        self.train_dataset = COCODataset(path=self.path,
                                         annotation_file=self.annotation_file_dict["train"],
                                         image_folder=self.image_folder,                                          
                                         fill_background=self.fill_background,
                                         id2label=self.id2label,
                                         transforms=self.train_transforms  
                                         )

        self.val_dataset = COCODataset(path=self.path,
                                       annotation_file=self.annotation_file_dict["val"],
                                       image_folder=self.image_folder,
                                       fill_background=self.fill_background,
                                       id2label=self.id2label)

        self.test_dataset = COCODataset(path=self.path,
                                        annotation_file=self.annotation_file_dict["test"],
                                        image_folder=self.image_folder,
                                        fill_background=self.fill_background,id2label=self.id2label)
                                        

    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                           batch_size=self.batch_size, 
                           shuffle=True, 
                           num_workers=self.num_workers, 
                           collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn)

    
    def collate_fn(self,batch):
        inputs = list(zip(*batch))        
        images=inputs[0]
        segmentation_maps=inputs[1]
        batch = self.preprocessor(
            images,
            segmentation_maps=segmentation_maps,
            size=self.img_size,
            return_tensors="pt",
            ignore_index = self.ignore_idx )
        batch["original_images"] = images
        batch["original_segmentation_maps"] = segmentation_maps
        return batch