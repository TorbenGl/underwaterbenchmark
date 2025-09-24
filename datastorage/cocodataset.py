from pycocotools.coco import COCO
import torch
from PIL import Image
from torchvision import tv_tensors
import torchvision.transforms.v2 as transforms
import numpy as np
from transformers import( Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor)
import torch.nn.functional as F

class COCODataset(torch.utils.data.Dataset):
    def __init__(self,path,annotation_file,image_folder,id2label,fill_background = True, transforms = None):
        self.path = path
        self.image_folder = image_folder
        self.fill_background = fill_background
        self.id2label = id2label
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.transforms = transforms
                    
              
        self.coco = COCO(self.path+annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        if self.fill_background:
          if "Background" not in self.label2id:
              self.fill_id = len(self.coco.cats)
          else:
            self.fill_id = self.label2id["Background"]
          print(f"Fill background with id {self.fill_id}")  
        
    def __getitem__(self, index: int):
        img_id = self.ids[index]
        img = self.path+self.image_folder+self.coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(img).convert("RGB")
        np_image=np.array(img)
        np_image = np_image.transpose(2,0,1)        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)        
        coco_annotation = self.coco.loadAnns(ann_ids)
        if self.fill_background:
          target = self.fill_id*np.ones((np_image.shape[-2], np_image.shape[-1]))
        else: 
            target = np.zeros((np_image.shape[-2], np_image.shape[-1]))            
        for ann in coco_annotation:
            mask = self.coco.annToMask(ann)
            target[mask == 1] = ann['category_id']
            
        
        return np_image, target
        
    def __len__(self):
        return len(self.ids)