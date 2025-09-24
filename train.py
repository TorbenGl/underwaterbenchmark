import lightning
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")

from models.mask2former import Mask2FormerFinetuner

from datastorage.cocodatamodule import CocoLightningDataModule

import configs.runconfig as config  

from transformers import Mask2FormerImageProcessor

if __name__=="__main__":

    preprocessor = Mask2FormerImageProcessor(**config.PREPROCESSOR_CONFIG,ignore_index=config.IGNORE_IDX)
    data_module = CocoLightningDataModule(path=config.DATASET_DIR,  
                                          image_folder=config.IMAGE_FOLDER, 
                                          annotation_file_dict=config.ANNOTATION_FILE_DICT, 
                                          fill_background=config.FILL_BACKGROUND, 
                                          devices=config.DEVICES, 
                                          batch_size=config.BATCH_SIZE, 
                                          num_workers=config.NUM_WORKERS, 
                                          img_size=config.IMG_SIZE,
                                          id2label=config.ID2LABEL, 
                                          ignore_idx=config.IGNORE_IDX, 
                                          preprocessor=preprocessor)
    model=Mask2FormerFinetuner(config.ID2LABEL, 
                               config.LEARNING_RATE, 
                               config.CHECKPOINTNAME,
                               config.IGNORE_IDX, 
                               preprocessor ,
                               max_epochs=config.EPOCHS,
                               freeze_encoder=config.FREEZE_ENCODER
                               )

    accelerator = 'cpu' if config.DEVICES[0] == "cpu" else 'cuda'

    trainer = lightning.Trainer(
        logger=config.LOGGER,
        accelerator=accelerator,
        devices=config.DEVICES,
        strategy="ddp",
        callbacks=[config.CHECKPOINT_CALLBACK],
        max_epochs=config.EPOCHS
    )
    print("Training starts!!")
    trainer.fit(model,data_module)
    print("saving model!")
    trainer.save_checkpoint("mask2former.ckpt")