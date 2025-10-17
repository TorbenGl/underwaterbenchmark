import lightning
import torch
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor
from torch import nn
import time
import json 
import numpy as np
from transformers import( Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor)
import torchmetrics

class Mask2FormerFinetuner(lightning.LightningModule):

    def __init__(self, id2label, lr, checkpointname, preprocessor, max_epochs ,freeze_encoder=True , encoder_lr_factor=0.1, batch_size=1):
        super(Mask2FormerFinetuner, self).__init__()
        self.id2label = id2label
        self.batch_size = batch_size        
        self.lr = lr
        print(f"max_epochs: {max_epochs}")
        self.max_epochs = max_epochs 
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpointname,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,            
        )       
        self.preprocessor = preprocessor
        self.freeze_encoder = freeze_encoder
        self.encoder_lr_factor = encoder_lr_factor
        if freeze_encoder:
          print("Freezing encoder")
          for param in self.model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False   
        self.save_hyperparameters() 
        
    def forward(self, pixel_values, pixel_mask=None, mask_labels=None, class_labels=None):
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels,pixel_mask=pixel_mask)
        
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        batch['pixel_values'] = batch['pixel_values'].to(device)
        batch['mask_labels'] = [label.to(device) for label in batch['mask_labels']]
        batch['class_labels'] = [label.to(device) for label in batch['class_labels']]
        batch['pixel_mask'] = batch['pixel_mask'].to(device)
        return batch

    def on_train_start(self):
        self.start_time = time.time()
        self.model.train()

    def on_train_end(self):
        total_time = time.time() - self.start_time
        metrics = {'final_epoch': self.current_epoch, 'training_time': total_time}
        self.model.train(False)  # Set model to evaluation mode        

    def training_step(self, batch, batch_idx):        
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
            pixel_mask=batch["pixel_mask"]
        )
        loss = outputs.loss
        self.log("trainLoss", loss, sync_dist=True,  batch_size=self.batch_size )
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        self.log("valLoss", loss, sync_dist=True,  batch_size=self.batch_size, on_epoch=True,logger=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=True, batch_size=self.batch_size, on_epoch=True, logger=True, prog_bar=True)
        return loss
    
    def on_validation_start(self):
        if self.model.training:
            self.model.train(False)

    def on_validation_epoch_end(self):
        self.model.train()
  
        
    def configure_optimizers(self): 
        # If encoder is frozen, optimize only the unfrozen parameters
        if self.freeze_encoder:
            optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
            print("Freezed encoder, using same LR for all params") 
        # If encoder is not frozen, use a lower learning rate for the encoder       
        else:
            print("Using lower LR for encoder")
            encoder_params = [p for p in self.model.model.pixel_level_module.encoder.parameters() if p.requires_grad]
            other_params = [p for n, p in self.model.named_parameters() if  not n.startswith('model.pixel_level_module.encoder')]
            optimizer = torch.optim.AdamW([
                {'params': encoder_params, 'lr': self.lr * self.encoder_lr_factor},
                {'params': other_params, 'lr': self.lr}
            ])
        scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-6),'monitor': 'valLoss'}
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    