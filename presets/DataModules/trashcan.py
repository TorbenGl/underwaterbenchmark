"""
TrashCan Dataset Presets.

Two variants for underwater trash detection:
1. trashcan_instance - 22 instance classes (specific trash types)
2. trashcan_material - 16 material classes (trash by material type)

Paper: https://arxiv.org/abs/2007.08097
Source: https://conservancy.umn.edu/handle/11299/214865
"""

from presets.DataModules import COCODatamodulePreset, register_dataset


# =============================================================================
# CLASS DEFINITIONS
# =============================================================================

# 22 instance classes (with background at 0 due to fill_background=True)
TRASHCAN_INSTANCE_ID2LABEL = {
    0: "background",
    1: "rov",
    2: "plant",
    3: "animal_fish",
    4: "animal_starfish",
    5: "animal_shells",
    6: "animal_crab",
    7: "animal_eel",
    8: "animal_etc",
    9: "trash_clothing",
    10: "trash_pipe",
    11: "trash_bottle",
    12: "trash_bag",
    13: "trash_snack_wrapper",
    14: "trash_can",
    15: "trash_cup",
    16: "trash_container",
    17: "trash_unknown_instance",
    18: "trash_branch",
    19: "trash_wreckage",
    20: "trash_tarp",
    21: "trash_rope",
    22: "trash_net",
}

# 16 material classes (with background at 0 due to fill_background=True)
TRASHCAN_MATERIAL_ID2LABEL = {
    0: "background",
    1: "rov",
    2: "plant",
    3: "animal_fish",
    4: "animal_starfish",
    5: "animal_shells",
    6: "animal_crab",
    7: "animal_eel",
    8: "animal_etc",
    9: "trash_etc",
    10: "trash_fabric",
    11: "trash_fishing_gear",
    12: "trash_metal",
    13: "trash_paper",
    14: "trash_plastic",
    15: "trash_rubber",
    16: "trash_wood",
}

# label2id mappings
TRASHCAN_INSTANCE_LABEL2ID = {v: k for k, v in TRASHCAN_INSTANCE_ID2LABEL.items()}
TRASHCAN_MATERIAL_LABEL2ID = {v: k for k, v in TRASHCAN_MATERIAL_ID2LABEL.items()}


# =============================================================================
# DATASET PRESETS
# =============================================================================

@register_dataset
class TrashCanInstanceDataset(COCODatamodulePreset):
    """
    TrashCan dataset with 22 instance classes.

    Classes include:
        - ROV, plants, animals (fish, starfish, shells, crab, eel, etc)
        - Trash types: clothing, pipe, bottle, bag, snack_wrapper, can, cup,
          container, unknown_instance, branch, wreckage, tarp, rope, net

    Dataset structure expected:
        {DATA_ROOT}/trashcan/dataset/instance_version/
            train/
                *.jpg
            val/
                *.jpg
            instances_train_trashcan.json
            instances_val_trashcan.json
    """
    name = "trashcan_instance"
    data_subdir = "trashcan/dataset/instance_version"
    image_folder = {
        "train": "train",
        "val": "val",
        "test": "val",  # Using val images for test
    }
    annotation_files = {
        "train": "instances_train_trashcan.json",
        "val": "instances_val_trashcan.json",
        "test": "instances_val_trashcan.json",  # Using val as test
    }
    default_img_size = (360, 480)  # Height, Width
    ignore_index = None
    increase_idx = False
    fill_background = True
    num_classes = len(TRASHCAN_INSTANCE_ID2LABEL)
    id2label = TRASHCAN_INSTANCE_ID2LABEL
    label2id = TRASHCAN_INSTANCE_LABEL2ID


@register_dataset
class TrashCanMaterialDataset(COCODatamodulePreset):
    """
    TrashCan dataset with 16 material classes.

    Classes include:
        - ROV, plants, animals (fish, starfish, shells, crab, eel, etc)
        - Trash by material: etc, fabric, fishing_gear, metal, paper,
          plastic, rubber, wood

    Dataset structure expected:
        {DATA_ROOT}/trashcan/dataset/material_version/
            train/
                *.jpg
            val/
                *.jpg
            instances_train_trashcan.json
            instances_val_trashcan.json
    """
    name = "trashcan_material"
    data_subdir = "trashcan/dataset/material_version"
    image_folder = {
        "train": "train",
        "val": "val",
        "test": "val",  # Using val images for test
    }
    annotation_files = {
        "train": "instances_train_trashcan.json",
        "val": "instances_val_trashcan.json",
        "test": "instances_val_trashcan.json",  # Using val as test
    }
    default_img_size = (360, 480)   # Height, Width
    ignore_index = None
    increase_idx = False
    fill_background = True
    num_classes = len(TRASHCAN_MATERIAL_ID2LABEL)
    id2label = TRASHCAN_MATERIAL_ID2LABEL
    label2id = TRASHCAN_MATERIAL_LABEL2ID
