from pycocotools.coco import COCO
import torch
from PIL import Image
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
import numpy as np
from transformers import( Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor)
import torch.nn.functional as F

class COCODataset_Instance(torch.utils.data.Dataset):
    def __init__(self,path,annotation_file,image_folder,id2label, transforms = None, increment_classes = False):
        self.path = path
        self.image_folder = image_folder
        self.id2label = id2label
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.transforms = transforms
        
        self.increment_classes = int(increment_classes)
        
        self.coco = COCO(self.path+annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))        
        
    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img = self.path+self.image_folder+self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(img).convert("RGB")
        np_image=np.array(img)
        np_image = np_image.transpose(2,0,1)        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)        
        coco_annotation = self.coco.loadAnns(ann_ids)
        target_mask = np.ones((np_image.shape[-2],np_image.shape[-1]), dtype=bool) * self.fill_id if self.fill_background else np.zeros((np_image.shape[-2],np_image.shape[-1]), dtype=np.int32) 
        instance_id = 1
        target_classes = dict()
        for ann in coco_annotation:
            mask = self.coco.annToMask(ann)
            target_mask[mask==1] = instance_id 
            target_classes[instance_id] = ann['category_id'] + self.increment_classes  
            instance_id += 1
        return np_image, target_mask, target_classes
        
    def __len__(self):
        return len(self.ids)