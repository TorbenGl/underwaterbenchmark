import lightning
import torch
from transformers import Mask2FormerForUniversalSegmentation
from transformers import AutoImageProcessor
from torch import nn
import evaluate
import time
import json 
import numpy as np
from transformers import( Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor)
import configs.runconfig  as config
import torchmetrics

class Mask2FormerFinetuner(lightning.LightningModule):

    def __init__(self, id2label, lr, checkpointname, ignore_idx, preprocessor,max_epochs,freeze_encoder=True):
        super(Mask2FormerFinetuner, self).__init__()
        self.id2label = id2label
        self.lr = lr
        self.max_epochs = max_epochs 
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            checkpointname,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,            
        )
        
        self.ignore_idx = ignore_idx
        self.preprocessor = preprocessor
        if freeze_encoder:
          print("Freezing encoder")
          for param in self.model.model.pixel_level_module.encoder.parameters():
            param.requires_grad = False                

        self.save_hyperparameters() 
        evaluate.load
        self.val_mean_iou = evaluate.load("mean_iou")
       #self.test_map_metric = torchmetrics.detection.MAP(
       #    iou_type='segm',
       #    iou_thresholds=torch.tensor([0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]),
       #    class_metrics=True
       #)

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
        with open('mask2former_hyperparameters.json', 'w') as f:
            json.dump(metrics, f)

    def training_step(self, batch, batch_idx):        
        outputs = self(
            pixel_values=batch["pixel_values"],
            mask_labels=batch["mask_labels"],
            class_labels=batch["class_labels"],
            pixel_mask=batch["pixel_mask"]
        )
        loss = outputs.loss
        self.log("trainLoss", loss, sync_dist=True,  batch_size=config.BATCH_SIZE )
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            pixel_values=batch["pixel_values"],
            pixel_mask=batch["pixel_mask"],
            mask_labels=[labels for labels in batch["mask_labels"]],
            class_labels=[labels for labels in batch["class_labels"]],
        )
        loss = outputs.loss
        self.log("valLoss", loss, sync_dist=True,  batch_size=config.BATCH_SIZE, on_epoch=True,logger=True, prog_bar=True)
        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("learning_rate", lr, sync_dist=True, batch_size=config.BATCH_SIZE, on_epoch=True, logger=True, prog_bar=True)
        return loss
    

    def on_validation_start(self):
        if self.model.training:
            self.model.train(False)

    def on_validation_epoch_end(self):
        self.model.train()

    def test_step(self, batch, batch_idx):
        imgs, targets = batch      
        outputs = self(imgs,targets) 
        loss = outputs.loss
        self.log("valLoss", loss, sync_dist=True,  batch_size=config.BATCH_SIZE, on_epoch=True,logger=True, prog_bar=True)
        batch_predictions = [self.masks2predictions(o["pred_masks"], o["pred_classes"]) for o in outputs]
        batch_targets = targets



        #return(metrics)
        
   
       # imgs, masks, classes = batch
       # original_images = imgs
       # imgs = torch.stack([torch.tensor(i) for i in self.processor(imgs)["pixel_values"]])
       # outputs = self(
       #     pixel_values=imgs,
       #     mask_labels=masks,
       #     class_labels=classes,
       # )
       # 
       # ground_truth = batch["original_segmentation_maps"]
       # target_sizes = [(image.shape[1], image.shape[2]) for image in original_images]
       # # predict segmentation maps
       # predicted_segmentation_maps =self.processor.post_process_semantic_segmentation(outputs,target_sizes=target_sizes)
       # 
       # predictions=predicted_segmentation_maps[0].cpu().numpy()
       # 
       # # Calculate FN and FP
       # false_negatives = np.sum((predictions == 0) & (ground_truth[0] == 1))
       # false_positives = np.sum((predictions == 1) & (ground_truth[0] == 0))
       # 
       # # Total number of instances
       # total_instances = np.prod(predictions.shape)
       # 
       # # Calculate percentages
       # percentage_fn = (false_negatives / total_instances) 
       # percentage_fp = (false_positives / total_instances) 
       # 
       # # Optionally log loss here
       # metrics = self.test_mean_iou._compute(
       #     predictions=predictions,
       #     references=ground_truth[0],
       #     num_labels=self.num_classes,
       #     ignore_index=254,
       #     reduce_labels=False,
       # )
       # # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
       # per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
       # per_category_iou = metrics.pop("per_category_iou").tolist()
    #
       # # Re-define metrics dict to include per-category metrics directly
       # metrics = {
       #     'testLoss': loss, 
       #     "mean_iou": metrics["mean_iou"], 
       #     "mean_accuracy": metrics["mean_accuracy"],
       #     "False Negative": percentage_fn,
       #     "False Positive": percentage_fp,
       #     **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
       #     **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
       # }
       # for k,v in metrics.items():
       #     self.log(k,v,sync_dist=True, batch_size=config.BATCH_SIZE)
       # return(metrics)
        
    def configure_optimizers(self):
        
        # AdamW optimizer with specified learning rate
        optimizer = torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.lr)
      
        # ReduceLROnPlateau scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.max_epochs, eta_min=1e-6),            
            'monitor': 'valLoss'  # Metric to monitor for reducing learning rate
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    def masks2predictions(pred_masks, pred_classes):
    # pred_masks: [num_queries, H, W]
    # pred_classes: [num_queries]

        pred_masks = (pred_masks.sigmoid() > 0.5).float()  # binarize
        scores = pred_masks.flatten(1).mean(dim=1)         # mean mask probability as score

        return {
            "masks": pred_masks,
            "scores": scores,
            "labels": pred_classes
        }