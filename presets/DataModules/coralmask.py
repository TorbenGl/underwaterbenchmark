"""
CoralMask Dataset Preset.

Dataset for coral instance segmentation (converted to semantic segmentation).

Before using this preset, run the preparation script:
    python scripts/prepare_coralmask.py /path/to/CoralMask

This converts per-image JSON annotations to COCO format.
"""

from presets.DataModules import COCODatamodulePreset, register_dataset


# Binary segmentation: background (0) vs coral (1)
CORALMASK_ID2LABEL = {
    0: "background",
    1: "coral",
}

CORALMASK_LABEL2ID = {v: k for k, v in CORALMASK_ID2LABEL.items()}


@register_dataset
class CoralMaskDataset(COCODatamodulePreset):
    """
    CoralMask dataset for coral segmentation.

    Binary segmentation task:
        - Class 0: Background
        - Class 1: Coral

    Dataset structure expected:
        {DATA_ROOT}/coralmask/CoralMask/
            train/
                images/*.jpg
                jsons/*.json (per-image annotations - converted by prepare script)
            test/
                images/*.jpg
                jsons/*.json
            train_annotations.json  (created by prepare_coralmask.py)
            test_annotations.json   (created by prepare_coralmask.py)

    Note: Run `python scripts/prepare_coralmask.py` first to convert
          per-image JSONs to COCO format annotation files.
    """

    name = "coralmask"
    data_subdir = "coralmask/CoralMask"  # Relative to DATA_ROOT
    image_folder = {
        "train": "train/images",
        "val": "train/images",
        "test": "test/images",
    }
    annotation_files = {
        "train": "train_annotations.json",
        "val": "train_annotations.json",  # Using test as val (no separate val split)
        "test": "test_annotations.json",
    }
    default_img_size = (540, 960)  # Height, Width
    ignore_index = 255
    increase_idx = False
    fill_background = True  # Add background class at index 0
    num_classes = len(CORALMASK_ID2LABEL)
    id2label = CORALMASK_ID2LABEL
    label2id = CORALMASK_LABEL2ID
