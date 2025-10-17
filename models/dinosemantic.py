import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import lightning
import time


class DinoFinetuner(lightning.LightningModule):
    def __init__(self, id2label, lr, checkpointname, preprocessor, max_epochs ,freeze_encoder=True , encoder_lr_factor=0.1, batch_size=1, kernel_size=1):
        super().__init__()
        self.id2label = id2label
        self.lr = lr
        print(f"max_epochs: {max_epochs}")
        self.max_epochs = max_epochs 
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.preprocessor = preprocessor
        self.criterion = nn.CrossEntropyLoss()
        self.kernel_size = kernel_size

        self.encoder = timm.create_model(checkpointname, pretrained=True,features_only=True,)

        for param in self.encoder.parameters():
            print("freezing encoder")
            param.requires_grad = False
        feature_channels = self.encoder.feature_info.channels()[-1]      
        self.classifier = nn.Conv2d(feature_channels, self.num_classes, kernel_size=self.kernel_size)
        self.batch_size = batch_size
        self.encoder_lr_factor = encoder_lr_factor
        self.save_hyperparameters()

    def forward(self, x):
        input_h, input_w = x.shape[-2], x.shape[-1]        
        features = self.encoder(x)[-1]  # Get the last feature map        
        features = F.interpolate(features, size=(input_h, input_w), mode='bilinear', align_corners=False) # Upsample to input size        
        logits = self.classifier(features) # 1x1 convolution to get class logits        
        return logits
    
    def on_train_start(self):
        self.start_time = time.time()
    def on_train_end(self):
        total_time = time.time() - self.start_time
        metrics = {'final_epoch': self.current_epoch, 'training_time': total_time}
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
       if 'pixel_values' in batch and batch['pixel_values'] is not None:
        batch['pixel_values'] = batch['pixel_values'].to(device)
        if 'mask_labels' in batch and batch['mask_labels'] is not None:
            batch['mask_labels'] = batch['mask_labels'].to(device) 
        if 'class_labels' in batch and batch['class_labels'] is not None:
            batch['class_labels'] = [label.to(device) for label in batch['class_labels']]
        if 'pixel_mask' in batch and batch['pixel_mask'] is not None:
            batch['pixel_mask'] = batch['pixel_mask'].to(device)
        return batch

    def training_step(self, batch, batch_idx):        
        # Implement the training step
        x = batch["pixel_values"]
        y = batch["mask_labels"]       
        logits = self(x)  # Forward pass
        loss = self.criterion(logits, y)  # Compute
        self.log("trainLoss", loss, sync_dist=True,  batch_size=self.batch_size )
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch["pixel_values"]
        y = batch["mask_labels"]       
        logits = self(x)  # Forward pass
        loss = self.criterion(logits, y)  # Compute
        self.log("valLoss", loss, sync_dist=True,  batch_size=self.batch_size, on_epoch=True,logger=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=True, batch_size=self.batch_size, on_epoch=True, logger=True, prog_bar=True)
        return loss
        
    def configure_optimizers(self):        
        # AdamW optimizer with specified learning rate
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)      
        # ReduceLROnPlateau scheduler
        scheduler = {            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-6),            
                                 'monitor': 'valLoss'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}