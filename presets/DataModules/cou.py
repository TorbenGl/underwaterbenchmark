"""
COU (Coral Object Understanding) Dataset Preset.

Configuration extracted from trainjobs.sh for the underwater benchmark.
"""

from presets.DataModules import COCODatamodulePreset, register_dataset


# id2label with background at 0 (due to fill_background=True and increase_idx=True)
# Metal Rod class removed (had 0 annotations in dataset)
COU_ID2LABEL = {
    0: "background",
    1: "Unknown Instance",
    2: "Scissors",
    3: "Plastic Cup",
    4: "Fork",
    5: "Bottle",
    6: "Soda Can",
    7: "Case",
    8: "Plastic Bag",
    9: "Cup",
    10: "Goggles",
    11: "Flipper",
    12: "LoCo",
    13: "Aqua",
    14: "Pipe",
    15: "Snorkel",
    16: "Spoon",
    17: "Lure",
    18: "Screwdriver",
    19: "Car",
    20: "Tripod",
    21: "ROV",
    22: "Knife",
    23: "Dive Weight",
}

COU_LABEL2ID = {v: k for k, v in COU_ID2LABEL.items()}


@register_dataset
class COUDataset(COCODatamodulePreset):
    """
    COU (Coral Object Understanding) dataset for underwater semantic segmentation.

    Dataset structure expected:
        {DATA_ROOT}/cou/coco/coco/
            images/
                *.jpg / *.png
            train_annotations_no_metalrod.json
            val_annotations_no_metalrod.json
            test_annotations_no_metalrod.json

    Note: Metal Rod class was removed (had 0 annotations in dataset).
    Use scripts/prepare_cou.py to generate the _no_metalrod annotation files.

    Default configuration:
        - Image size: 540x960 (HxW)
        - Ignore index: 255
        - Background filling: enabled
        - Index increase: enabled (background = 0)
        - 24 classes (0-23, including background)
    """
    name = "cou"
    data_subdir = "cou/coco/coco"  # Relative to DATA_ROOT
    image_folder = "images/"
    annotation_files = {
        "train": "train_annotations_no_metalrod.json",
        "val": "val_annotations_no_metalrod.json",
        "test": "test_annotations_no_metalrod.json",
    }
    default_img_size =  (540, 960)  # Height, Width from trainjobs.sh
    ignore_index = 255
    increase_idx = True
    fill_background = True
    id2label = COU_ID2LABEL
    label2id = COU_LABEL2ID
