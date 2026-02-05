"""
L4S (Looking for Seagrass) Dataset Preset.

Dataset for underwater seagrass semantic segmentation.

Before using this preset, run the preparation script:
    python scripts/prepare_l4s.py /path/to/l4s/dataset

This converts the L4S JSON format to the ImageMaskPreset format.
"""

from presets.DataModules import ImageMaskDatamodulePreset, register_dataset


# Binary segmentation: background + seagrass
L4S_ID2LABEL = {
    0: "background",
    1: "seagrass",
}

L4S_LABEL2ID = {v: k for k, v in L4S_ID2LABEL.items()}


@register_dataset
class L4SDataset(ImageMaskDatamodulePreset):
    """
    L4S (Looking for Seagrass) dataset for underwater seagrass segmentation.

    Binary segmentation task:
        - Class 0: Background (water, sand, etc.)
        - Class 1: Seagrass

    Dataset structure expected:
        {DATA_ROOT}/l4s/
            images/
                00d01/
                    *.jpg
                01d02/
                    *.jpg
                ...
            ground-truth/
                00d01/
                    pm_*.png  (8-bit grayscale masks)
                ...
            train_samples.json  (created by prepare_l4s.py)
            val_samples.json    (created by prepare_l4s.py)
            test_samples.json   (created by prepare_l4s.py)

    Note: Run `python scripts/prepare_l4s.py` first to create the sample files.
    """

    name = "l4s"
    data_subdir = "l4s/dataset"  # Relative to DATA_ROOT
    train_annotation_file = "train_samples.json"
    val_annotation_file = "val_samples.json"
    test_annotation_file = "test_samples.json"
    default_img_size =  (540, 960) # Height, Width (downscaled from 1080x1920)
    num_classes = 2  # Background + Seagrass
    id2label = L4S_ID2LABEL
    label2id = L4S_LABEL2ID
    rgb_mask = False  # Masks are grayscale
    background_color = (0, 0, 0)
    increase_idx = False  # Normalize binary masks: 0/255 -> 0/1
    value_to_class = {0: 1, 255: 0}  # Map 0 -> 0 (background), 255 -> 1 (seagrass)
