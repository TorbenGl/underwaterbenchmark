"""
LIACi (Learnable Image-based Autonomous Crack Inspection) Dataset Preset.

Dataset for underwater ship hull inspection with semantic segmentation.

Before using this preset, run the preparation script:
    python scripts/prepare_liaci.py /path/to/LIACi_dataset_pretty

This creates the split annotation files (coco-labels_train.json, coco-labels_test.json).
"""

from presets.DataModules import COCODatamodulePreset, register_dataset


# id2label with background at 0 (due to fill_background=True)
# Categories are remapped to contiguous indices 1-10
LIACI_ID2LABEL = {
    0: "background",
    1: "sea_chest_grating",
    2: "paint_peel",
    3: "over_board_valve",
    4: "defect",
    5: "corrosion",
    6: "propeller",
    7: "anode",
    8: "bilge_keel",
    9: "marine_growth",
    10: "ship_hull",
}

LIACI_LABEL2ID = {v: k for k, v in LIACI_ID2LABEL.items()}


@register_dataset
class LIACiDataset(COCODatamodulePreset):
    """
    LIACi dataset for underwater ship hull inspection.

    Classes include:
        - anode
        - bilge_keel
        - corrosion
        - defect
        - marine_growth
        - over_board_valves
        - paint_peel
        - propeller
        - sea_chest_grating
        - ship_hull

    Dataset structure expected:
        {DATA_ROOT}/liaci/
            images/
                image_0001.jpg
                image_0002.jpg
                ...
            coco-labels_train.json  (created by prepare_liaci.py)
            coco-labels_test.json   (created by prepare_liaci.py)

    Note: Run `python scripts/prepare_liaci.py` first to create the split files.
    """

    name = "liaci"
    data_subdir = "liaci/LIACi_dataset_pretty"  # Relative to DATA_ROOT
    image_folder = "images"
    annotation_files = {
        "train": "coco-labels_train.json",
        "val": "coco-labels_train.json",  # Using train as val
        "test": "coco-labels_test.json",
    }
    default_img_size =  (1080, 1920)  # Height, Width (HD aspect ratio, downscaled)
    ignore_index = None
    increase_idx = True  # Background = 0, classes start at 1
    fill_background = True
    id2label = LIACI_ID2LABEL
    label2id = LIACI_LABEL2ID
