"""
SeaClear Marine Debris Dataset Presets.

Three class configurations for underwater marine debris detection:
1. seaclear_base - Original 40 classes
2. seaclear_mat - 11 material classes (plastic, metal, glass, etc.)
3. seaclear_sup - 5 superclasses (ANIMAL, DEBRIS, ROV, NATURAL, UNKNOWN)

Before using these presets, run the preparation script:
    python scripts/create_seaclear_splits.py /path/to/seaclear/dataset

This creates the train/test split files in the 'splits' subdirectory.
"""

from presets.DataModules import COCODatamodulePreset, register_dataset


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

# 40 base classes (from dataset.json categories)
BASE_ID2LABEL = {
    0: "background",
    1: "can_metal",
    2: "bottle_plastic",
    3: "bag_plastic",
    4: "cup_plastic",
    5: "container_plastic",
    6: "bottle_glass",
    7: "jar_glass",
    8: "pipe_plastic",
    9: "sanitaries_plastic",
    10: "snack_wrapper_plastic",
    11: "tarp_plastic",
    12: "net_plastic",
    13: "rope_plastic",
    14: "lid_plastic",
    15: "tire_rubber",
    16: "boot_rubber",
    17: "rope_fiber",
    18: "clothing_fiber",
    19: "cup_ceramic",
    20: "wreckage_metal",
    21: "cable_metal",
    22: "tube_cement",
    23: "brick_clay",
    24: "container_middle_size_metal",
    25: "cardboard_paper",
    26: "snack_wrapper_paper",
    27: "branch_wood",
    28: "furniture_wood",
    29: "plant",
    30: "rov_cable",
    31: "rov_tortuga",
    32: "rov_bluerov",
    33: "rov_vehicle_leg",
    34: "unknown_instance",
    35: "animal_urchin",
    36: "animal_fish",
    37: "animal_shells",
    38: "animal_sponge",
    39: "animal_starfish",
    40: "animal_etc",
}

# 11 material classes
MATERIAL_ID2LABEL = {       
    0: "background",
    1: "animal",
    2: "cement",
    3: "ceramic",
    4: "fiber",
    5: "glass",
    6: "metal",
    7: "paper",
    8: "plastic",
    9: "rubber",
    10: "unknown",
    11: "wood",
}

# 5 superclasses
SUPERCLASS_ID2LABEL = {
    0: "background",
    1: "ANIMAL",
    2: "DEBRIS",
    3: "NATURAL",
    4: "ROV",
    5: "UNKNOWN",
}

# label2id mappings
BASE_LABEL2ID = {v: k for k, v in BASE_ID2LABEL.items()}
MATERIAL_LABEL2ID = {v: k for k, v in MATERIAL_ID2LABEL.items()}
SUPERCLASS_LABEL2ID = {v: k for k, v in SUPERCLASS_ID2LABEL.items()}


# =============================================================================
# DATASET PRESETS
# =============================================================================

@register_dataset
class SeaClearBaseDataset(COCODatamodulePreset):
    """
    SeaClear dataset with original 40 classes.

    Dataset structure expected:
        {DATA_ROOT}/seaclear/Seaclear Marine Debris Dataset/
            splits/
                base_train.json
                base_test.json
            Bistrina/
                Bluerobotics HD/
                Paralenz Vaquita Gen 2/
                SIP-E323CV/
            Jakljan/
            Lokrum/
            Marseille/
            Slano/

    Note: Run `python scripts/create_seaclear_splits.py` first to create split files.
    """
    name = "seaclear_base"
    data_subdir = "seaclear/Seaclear Marine Debris Dataset"
    image_folder = ""  # Images are in site/dive/ structure
    annotation_files = {
        "train": "splits/base_train.json",
        "val": "splits/base_train.json",  # Using test as val
        "test": "splits/base_test.json",
    }
    default_img_size = (540, 960)  # Height, Width (downscaled from 1080x1920)
    ignore_index = None
    increase_idx = True
    fill_background = True
    num_classes = 40
    id2label = BASE_ID2LABEL
    label2id = BASE_LABEL2ID


@register_dataset
class SeaClearMaterialDataset(COCODatamodulePreset):
    """
    SeaClear dataset with 11 material classes.

    Classes: animal, cement, ceramic, fiber, glass, metal, paper, plastic, rubber, unknown, wood

    Dataset structure expected:
        {DATA_ROOT}/seaclear/Seaclear Marine Debris Dataset/
            splits/
                material_train.json
                material_test.json
            ...

    Note: Run `python scripts/create_seaclear_splits.py` first to create split files.
    """
    name = "seaclear_mat"
    data_subdir = "seaclear/Seaclear Marine Debris Dataset"
    image_folder = ""
    annotation_files = {
        "train": "splits/material_train.json",
        "val": "splits/material_train.json",
        "test": "splits/material_test.json",
    }
    default_img_size = (540, 960)
    ignore_index = None
    increase_idx = False
    fill_background = True
    num_classes = len(MATERIAL_ID2LABEL)
    id2label = MATERIAL_ID2LABEL
    label2id = MATERIAL_LABEL2ID


@register_dataset
class SeaClearSuperclassDataset(COCODatamodulePreset):
    """
    SeaClear dataset with 5 superclasses.

    Classes: ANIMAL, DEBRIS, NATURAL, ROV, UNKNOWN

    Dataset structure expected:
        {DATA_ROOT}/seaclear/Seaclear Marine Debris Dataset/
            splits/
                superclass_train.json
                superclass_test.json
            ...

    Note: Run `python scripts/create_seaclear_splits.py` first to create split files.
    """
    name = "seaclear_sup"
    data_subdir = "seaclear/Seaclear Marine Debris Dataset"
    image_folder = ""
    annotation_files = {
        "train": "splits/superclass_train.json",
        "val": "splits/superclass_train.json",
        "test": "splits/superclass_test.json",
    }
    default_img_size = (540, 960)
    ignore_index = None
    increase_idx = False
    fill_background = True
    num_classes = len(SUPERCLASS_ID2LABEL)
    id2label = SUPERCLASS_ID2LABEL
    label2id = SUPERCLASS_LABEL2ID
