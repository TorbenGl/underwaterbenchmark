import lightning
import torch
torch.manual_seed(1)
torch.set_float32_matmul_precision("medium")

from models.dinosemantic import DinoFinetuner

from  datastorage.cocodatamodule_semantic import CocoLightningDataModule_Semantic

import configs.dino_runconfg as config

from training.dinopreprocessor import DinoPreprocessor

if __name__=="__main__":

    preprocessor = DinoPreprocessor(config.PREPROCESSOR_CONFIG,ignore_index=config.IGNORE_IDX)
    data_module = CocoLightningDataModule_Semantic(path=config.DATASET_DIR,  
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
    
    model=DinoFinetuner(config.ID2LABEL, 
                               config.LEARNING_RATE, 
                               config.CHECKPOINTNAME,                                
                               preprocessor ,
                               max_epochs=config.EPOCHS,
                               freeze_encoder=config.FREEZE_ENCODER,
                                batch_size=config.BATCH_SIZE,
                                encoder_lr_factor=config.ENCODER_LEARNING_RATE_FACTOR,
                                kernel_size=config.KERNEL_SIZE
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