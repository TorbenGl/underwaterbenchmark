"""
Coralscapes Dataset Preset.

HuggingFace dataset for underwater coral reef semantic segmentation.
Supports both HuggingFace Hub and local loading.
"""

from presets.DataModules import HuggingFaceDatamodulePreset, register_dataset


# 38 classes for coral reef semantic segmentation
# NOTE: Remapped via prepare_coralscapes.py script:
#   - background (13) + dark (14) merged to class 0
#   - classes 15-39 shifted down by 2
CORALSCAPES_ID2LABEL = {
    0: "background",  # Merged: original background (13) + dark (14)
    1: "seagrass",
    2: "trash",
    3: "other coral dead",
    4: "other coral bleached",
    5: "sand",
    6: "other coral alive",
    7: "human",
    8: "transect tools",
    9: "fish",
    10: "algae covered substrate",
    11: "other animal",
    12: "unknown hard substrate",
    13: "transect line",
    14: "massive/meandering bleached",
    15: "massive/meandering alive",
    16: "rubble",
    17: "branching bleached",
    18: "branching dead",
    19: "millepora",
    20: "branching alive",
    21: "massive/meandering dead",
    22: "clam",
    23: "acropora alive",
    24: "sea cucumber",
    25: "turbinaria",
    26: "table acropora alive",
    27: "sponge",
    28: "anemone",
    29: "pocillopora alive",
    30: "table acropora dead",
    31: "meandering bleached",
    32: "stylophora alive",
    33: "sea urchin",
    34: "meandering alive",
    35: "meandering dead",
    36: "crown of thorn",
    37: "dead clam",
}

CORALSCAPES_LABEL2ID = {v: k for k, v in CORALSCAPES_ID2LABEL.items()}


@register_dataset
class CoralscapesDataset(HuggingFaceDatamodulePreset):
    """
    Coralscapes dataset for underwater coral reef semantic segmentation.

    Supports two loading modes:
    1. HuggingFace Hub: Uses hf_dataset_name="EPFL-ECEO/coralscapes"
    2. Local: Uses data_subdir="coralscapes" (relative to DATA_ROOT)

    Dataset structure for local loading:
        {DATA_ROOT}/coralscapes/
            train/
            validation/
            test/
            dataset_info.json (optional)

    Default configuration:
        - Image size: 512x512 (HxW)
        - Image column: "image"
        - Mask column: "label"
        - Splits: train, validation, test
    """
    name = "coralscapes"

    # Local loading: data_subdir relative to DATA_ROOT
    # Set to None and use hf_dataset_name for Hub loading instead
    data_subdir = "coralscapes"

    # HuggingFace Hub loading (only used if data_subdir is None)
    # hf_dataset_name = "EPFL-ECEO/coralscapes"

    image_column = "image"
    mask_column = "label"
    split_mapping = {
        "train": "train",
        "val": "validation",
        "test": "test",
    }
    default_img_size = (512, 1024)
    ignore_index = 255
    trust_remote_code = True
    num_classes = 38
    id2label = CORALSCAPES_ID2LABEL
    label2id = CORALSCAPES_LABEL2ID
