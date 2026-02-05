"""
SUIM (Semantic Segmentation of Underwater Imagery) Dataset Preset.

Dataset for underwater semantic segmentation with 8 classes.

Before using this preset, run the preparation script:
    python scripts/prepare_suim.py /path/to/suim

This creates the train/val/test split annotation files.
"""

from presets.DataModules import ImageMaskDatamodulePreset, register_dataset


# 8 classes for underwater semantic segmentation
# Masks are RGB color-coded, values are binary (0 or 255) per channel
SUIM_ID2LABEL = {
    0: "Background waterbody",
    1: "Human divers",
    2: "Plants/sea-grass",
    3: "Wrecks/ruins",
    4: "Robots/instruments",
    5: "Reefs and invertebrates",
    6: "Fish and vertebrates",
    7: "Sand/sea-floor",
}

# RGB color to class index mapping
# Colors follow binary encoding: R=4, G=2, B=1
SUIM_COLOR_TO_CLASS = {
    (0, 0, 0): 0,        # 000 - Background waterbody (BW)
    (0, 0, 255): 1,      # 001 - Human divers (HD)
    (0, 255, 0): 2,      # 010 - Plants/sea-grass (PF)
    (0, 255, 255): 3,    # 011 - Wrecks/ruins (WR)
    (255, 0, 0): 4,      # 100 - Robots/instruments (RO)
    (255, 0, 255): 5,    # 101 - Reefs and invertebrates (RI)
    (255, 255, 0): 6,    # 110 - Fish and vertebrates (FV)
    (255, 255, 255): 7,  # 111 - Sand/sea-floor (SR)
}

SUIM_LABEL2ID = {v: k for k, v in SUIM_ID2LABEL.items()}


@register_dataset
class SUIMDataset(ImageMaskDatamodulePreset):
    """
    SUIM dataset for underwater semantic segmentation.

    8 classes:
        - Background waterbody (BW)
        - Human divers (HD)
        - Plants/sea-grass (PF)
        - Wrecks/ruins (WR)
        - Robots/instruments (RO)
        - Reefs and invertebrates (RI)
        - Fish and vertebrates (FV)
        - Sand/sea-floor (SR)

    Dataset structure expected:
        {DATA_ROOT}/suim/
            train_val/
                images/*.jpg
                masks/*.bmp (RGB color-coded)
            TEST/
                images/*.jpg
                masks/*.bmp
            train_samples.json  (created by prepare_suim.py)
            val_samples.json    (created by prepare_suim.py)
            test_samples.json   (created by prepare_suim.py)

    Note: Run `python scripts/prepare_suim.py` first to create the sample files.
    """

    name = "suim"
    data_subdir = "suim"  # Relative to DATA_ROOT
    train_annotation_file = "train_samples.json"
    val_annotation_file = "val_samples.json"
    test_annotation_file = "test_samples.json"
    default_img_size =  (480, 640)  # Height, Width (original size)
    num_classes = 8
    id2label = SUIM_ID2LABEL
    label2id = SUIM_LABEL2ID
    rgb_mask = True  # Masks are RGB color-coded
    color_to_class = SUIM_COLOR_TO_CLASS
    background_color = (0, 0, 0)
    increase_idx = False
