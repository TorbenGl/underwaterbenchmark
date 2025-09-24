from lightning.pytorch.callbacks import ModelCheckpoint
import sys
import os
from lightning.pytorch.loggers import WandbLogger


wandb_logger = WandbLogger(log_model="all")
#Training hyperparmeters
CHECKPOINTNAME = "ClownRat/mask2former-resnet-50-coco-instance"
LEARNING_RATE=0.0001
EPOCHS=50
PRECISION='16-mixed'
DEVICES=[0]
IGNORE_IDX=-1
CHECKPOINT_CALLBACK = ModelCheckpoint(save_top_k=1, 
                                      monitor="valLoss", 
                                      every_n_epochs=2,  # Save the model at every epoch 
                                      save_on_train_epoch_end=True,
                                      dirpath="/home/ida01/tglobisch/checkpoints/cou" # Ensure saving happens at the end of a training epoch
                                     )
LOGGER = WandbLogger(log_model="all")
FREEZE_ENCODER = False

#Dataset
DATASET_DIR="/mnt/scratch/tglobisch/datasets/cou/cou/coco/coco/"
IMAGE_FOLDER = "images/"
ANNOTATION_FILE_DICT = {
    "train": "train_annotations.json",
    "val": "val_annotations.json",
    "test": "test_annotations.json"
}
FILL_BACKGROUND = True
NUM_WORKERS=5
BATCH_SIZE=20
ID2LABEL={0: 'Unknown Instance',
 1: 'Scissors',
 2: 'Plastic Cup',
 3: 'Metal Rod',
 4: 'Fork',
 5: 'Bottle',
 6: 'Soda Can',
 7: 'Case',
 8: 'Plastic Bag',
 9: 'Cup',
 10: 'Goggles',
 11: 'Flipper',
 12: 'LoCo',
 13: 'Aqua',
 14: 'Pipe',
 15: 'Snorkel',
 16: 'Spoon',
 17: 'Lure',
 18: 'Screwdriver',
 19: 'Car',
 20: 'Tripod',
 21: 'ROV',
 22: 'Knife',
 23: 'Dive Weight',
 24: 'Background',
 }
IMG_SIZE = (272, 480)
PREPROCESSOR_CONFIG = {"do_normalize": True,
                "do_rescale": True,
                  "do_resize": True,
                    "image_mean": [
                      0.48500001430511475,
                      0.4560000002384186,
                      0.4059999883174896
                    ],
                    "image_processor_type": "Mask2FormerImageProcessor",
                    "image_std": [
                      0.2290000021457672,
                      0.2239999920129776,
                      0.22499999403953552
                    ],
                    "num_labels": 24,
                    "reduce_labels": False,
                    "resample": 2,
                    "rescale_factor": 0.00392156862745098,
                    "size": {
                      "height": 1080,
                      "width": 1920
                    },
                    "size_divisor": 16
                    }  