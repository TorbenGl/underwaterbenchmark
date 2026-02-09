"""
USOD10K (Underwater Salient Object Detection 10K) Dataset Preset.

Dataset for underwater salient object detection (binary segmentation).

Before using this preset, run the preparation script:
    python scripts/prepare_usod10k.py /path/to/USOD10k/USOD10k

This creates the train/val/test annotation files.
"""

from presets.DataModules import ImageMaskDatamodulePreset, register_dataset


# Binary segmentation: background (0) vs salient object (1)
USOD10K_ID2LABEL = {
    0: "background",
    1: "salient_object",
}

USOD10K_LABEL2ID = {v: k for k, v in USOD10K_ID2LABEL.items()}


@register_dataset
class USOD10KDataset(ImageMaskDatamodulePreset):
    """
    USOD10K dataset for underwater salient object detection.

    Binary segmentation task:
        - Class 0: Background
        - Class 1: Salient object (underwater objects of interest)

    Dataset structure expected:
        {DATA_ROOT}/uso10k/USOD10k/USOD10k/
            TR/
                RGB/*.png (images)
                GT/*.png (saliency masks, grayscale 0-255)
            VAL/
                RGB/*.png
                GT/*.png
            TE/
                RGB/*.png
                GT/*.png
            train_samples.json  (created by prepare_usod10k.py)
            val_samples.json    (created by prepare_usod10k.py)
            test_samples.json   (created by prepare_usod10k.py)

    Note: Original masks are saliency maps (0-255).
          Values >= 128 are treated as salient objects (class 1).
    """

    name = "usod10k"
    data_subdir = "uso10k/USOD10k/USOD10k"  # Relative to DATA_ROOT
    train_annotation_file = "train_samples.json"
    val_annotation_file = "val_samples.json"
    test_annotation_file = "test_samples.json"
    default_img_size = (540, 960) # Height, Width
    num_classes = 2
    id2label = USOD10K_ID2LABEL
    label2id = USOD10K_LABEL2ID
    rgb_mask = False  # Masks are grayscale (stored as RGB but same values)
    background_color = (0, 0, 0)
    increase_idx = False
    # Threshold saliency maps: values < 128 -> background (0), values >= 128 -> salient (1)
    # This requires custom handling in the dataset - for now use grayscale directly
