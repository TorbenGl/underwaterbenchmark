from lightning.pytorch.callbacks import ModelCheckpoint
import sys
import os
from lightning.pytorch.loggers import WandbLogger

# Logging
wandb_logger = WandbLogger(log_model="all")
LOGGER = wandb_logger

# Define Checkpoint Name
CHECKPOINTNAME = "timm/vit_base_patch16_dinov3.lvd1689m"
#Training hyperparmeters
LEARNING_RATE=0.0001
ENCODER_LEARNING_RATE_FACTOR=0.1

EPOCHS=50
PRECISION='16-mixed'
DEVICES=[0]
IGNORE_IDX=-1
CHECKPOINT_CALLBACK = ModelCheckpoint(save_top_k=1, 
                                      monitor="valLoss", 
                                      every_n_epochs=1,  # Save the model at every epoch 
                                      save_on_train_epoch_end=True,
                                      dirpath="/home/ida01/tglobisch/checkpoints/cou" # Ensure saving happens at the end of a training epoch
                                     )

FREEZE_ENCODER = True
KERNEL_SIZE=1
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
BATCH_SIZE=15

# Class Increment
INCREMENT_CLASSES = True  # Whether to use class increment or not
# If True, the dataset classes will be incremented by 1 to include background class at index 0

# Mapping from COCO category IDs AFTER Increament
ID2LABEL={
    0: 'Background',
    1: 'Unknown Instance',
    2: 'Scissors',
    3: 'Plastic Cup',
    4: 'Metal Rod',
    5: 'Fork',
    6: 'Bottle',
    7: 'Soda Can',
    8: 'Case',
    9: 'Plastic Bag',
    10: 'Cup',
    11: 'Goggles',
    12: 'Flipper',
    13: 'LoCo',
    14: 'Aqua',
    15: 'Pipe',
    16: 'Snorkel',
    17: 'Spoon',
    18: 'Lure',
    19: 'Screwdriver',
    20: 'Car',
    21: 'Tripod',
    22: 'ROV',
    23: 'Knife',
    24: 'Dive Weight',
    }




# Preprocessor config
IMG_SIZE = (272, 480)  # Height, Width
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
                    "size": {
                      "height": 272,
                      "width": 480
                    },
                    "size_divisor": 16
                    }  