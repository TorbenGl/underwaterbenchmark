"""
DeepFish Segmentation Dataset Preset.

Dataset for underwater fish binary segmentation.

Before using this preset, run the preparation script:
    python scripts/prepare_deepfish.py /path/to/DeepFish/Segmentation

This converts the DeepFish CSV splits to the ImageMaskPreset format.
"""

from presets.DataModules import ImageMaskDatamodulePreset, register_dataset


# Binary segmentation: background + fish
DEEPFISH_ID2LABEL = {
    0: "background",
    1: "fish",
}

DEEPFISH_LABEL2ID = {v: k for k, v in DEEPFISH_ID2LABEL.items()}


@register_dataset
class DeepFishDataset(ImageMaskDatamodulePreset):
    """
    DeepFish dataset for underwater fish segmentation.

    Binary segmentation task:
        - Class 0: Background (water, substrate, etc.)
        - Class 1: Fish

    Dataset structure expected:
        {DATA_ROOT}/deepfish/DeepFish/Segmentation/
            images/
                valid/*.jpg   (frames with fish)
                empty/*.jpg   (frames without fish)
            masks/
                valid/*.png   (grayscale: 0=background, 255=fish)
                empty/*.png   (all background)
            train_samples.json  (created by prepare_deepfish.py)
            val_samples.json    (created by prepare_deepfish.py)
            test_samples.json   (created by prepare_deepfish.py)

    Note: Run `python scripts/prepare_deepfish.py` first to create the sample files.
    """

    name = "deepfish"
    data_subdir = "deepfish/DeepFish/Segmentation"
    train_annotation_file = "train_samples.json"
    val_annotation_file = "val_samples.json"
    test_annotation_file = "test_samples.json"
    default_img_size = (540, 960)  # Height, Width (half of original 1080x1920)
    num_classes = 2  # Background + Fish
    id2label = DEEPFISH_ID2LABEL
    label2id = DEEPFISH_LABEL2ID
    rgb_mask = False  # Masks are grayscale
    background_color = (0, 0, 0)
    increase_idx = True  # Normalize binary masks: 0/255 -> 0/1
