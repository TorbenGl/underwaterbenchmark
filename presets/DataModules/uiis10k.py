"""
UIIS10K (Underwater Image Instance Segmentation 10K) Dataset Preset.

Dataset for underwater instance/semantic segmentation with 10 classes.

Paper: https://arxiv.org/abs/2202.01006
"""

from presets.DataModules import COCODatamodulePreset, register_dataset


# 10 classes (with background at 0 due to fill_background=True)
UIIS10K_ID2LABEL = {
    0: "background",
    1: "fish",
    2: "reptiles",
    3: "arthropoda",
    4: "corals",
    5: "mollusk",
    6: "plants",
    7: "ruins",
    8: "garbage",
    9: "human",
    10: "robots",
}

UIIS10K_LABEL2ID = {v: k for k, v in UIIS10K_ID2LABEL.items()}


@register_dataset
class UIIS10KDataset(COCODatamodulePreset):
    """
    UIIS10K dataset for underwater semantic segmentation.

    10 classes:
        - fish
        - reptiles
        - arthropoda (crustaceans, etc.)
        - corals
        - mollusk
        - plants
        - ruins
        - garbage
        - human
        - robots

    Dataset structure expected:
        {DATA_ROOT}/uiis10k/UIIS10K/
            img/
                train_*.jpg
                test_*.jpg
            annotations/
                multiclass_train.json
                multiclass_test.json
    """
    name = "uiis10k"
    data_subdir = "uiis10k/UIIS10K"
    image_folder = "img"
    annotation_files = {
        "train": "annotations/multiclass_train.json",
        "val": "annotations/multiclass_train.json",  # Using test as val
        "test": "annotations/multiclass_test.json",
    }
    default_img_size =  (480, 640)  # Height, Width
    ignore_index = None
    increase_idx = False
    fill_background = True
    num_classes = len(UIIS10K_ID2LABEL)
    id2label = UIIS10K_ID2LABEL
    label2id = UIIS10K_LABEL2ID
