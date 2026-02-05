#!/usr/bin/env python3
"""
Prepare SeaClear Marine Debris Dataset for training.

Creates reproducible train/test splits with three class configurations:
1. Base split - original 40 classes
2. Material split - classes mapped to 11 materials
3. Superclass split - classes mapped to 5 superclasses (ANIMAL, DEBRIS, ROV, NATURAL, UNKNOWN)

Image names are formatted as 'site/dive/imgname' for compatibility with COCO dataset loaders.

Usage:
    python scripts/create_seaclear_splits.py /path/to/seaclear/dataset

This creates in the 'splits' subdirectory:
    - base_train.json, base_test.json
    - material_train.json, material_test.json
    - superclass_train.json, superclass_test.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any

# ============================================================================
# CLASS MAPPINGS
# ============================================================================

CLASS_TO_SUPERCLASS = {
    # ANIMAL
    "animal_urchin": "ANIMAL",
    "animal_fish": "ANIMAL",
    "animal_shells": "ANIMAL",
    "animal_sponge": "ANIMAL",
    "animal_starfish": "ANIMAL",
    "animal_etc": "ANIMAL",
    # DEBRIS – plastic
    "bottle_plastic": "DEBRIS",
    "bag_plastic": "DEBRIS",
    "cup_plastic": "DEBRIS",
    "container_plastic": "DEBRIS",
    "pipe_plastic": "DEBRIS",
    "sanitaries_plastic": "DEBRIS",
    "snack_wrapper_plastic": "DEBRIS",
    "tarp_plastic": "DEBRIS",
    "net_plastic": "DEBRIS",
    "rope_plastic": "DEBRIS",
    "lid_plastic": "DEBRIS",
    # DEBRIS – metal
    "can_metal": "DEBRIS",
    "wreckage_metal": "DEBRIS",
    "cable_metal": "DEBRIS",
    "container_middle_size_metal": "DEBRIS",
    # DEBRIS – glass
    "bottle_glass": "DEBRIS",
    "jar_glass": "DEBRIS",
    # DEBRIS – rubber
    "tire_rubber": "DEBRIS",
    "boot_rubber": "DEBRIS",
    # DEBRIS – fiber
    "rope_fiber": "DEBRIS",
    "clothing_fiber": "DEBRIS",
    # DEBRIS – ceramic
    "cup_ceramic": "DEBRIS",
    # DEBRIS – cement / clay
    "tube_cement": "DEBRIS",
    "brick_clay": "DEBRIS",
    # DEBRIS – paper
    "cardboard_paper": "DEBRIS",
    "snack_wrapper_paper": "DEBRIS",
    # DEBRIS – wood
    "furniture_wood": "DEBRIS",
    # ROV
    "rov_cable": "ROV",
    "rov_tortuga": "ROV",
    "rov_bluerov": "ROV",
    "rov_vehicle_leg": "ROV",
    # NATURAL
    "plant": "NATURAL",
    "branch_wood": "NATURAL",
    # UNKNOWN
    "unknown_instance": "UNKNOWN",
}

CLASS_TO_MATERIAL = {
    "animal_urchin": "animal",
    "animal_fish": "animal",
    "animal_shells": "animal",
    "animal_sponge": "animal",
    "animal_starfish": "animal",
    "animal_etc": "animal",
    "bottle_plastic": "plastic",
    "bag_plastic": "plastic",
    "cup_plastic": "plastic",
    "container_plastic": "plastic",
    "pipe_plastic": "plastic",
    "sanitaries_plastic": "plastic",
    "snack_wrapper_plastic": "plastic",
    "tarp_plastic": "plastic",
    "net_plastic": "plastic",
    "rope_plastic": "plastic",
    "lid_plastic": "plastic",
    "can_metal": "metal",
    "wreckage_metal": "metal",
    "cable_metal": "metal",
    "container_middle_size_metal": "metal",
    "rov_vehicle_leg": "metal",
    "rov_bluerov": "metal",
    "rov_tortuga": "metal",
    "bottle_glass": "glass",
    "jar_glass": "glass",
    "tire_rubber": "rubber",
    "boot_rubber": "rubber",
    "rope_fiber": "fiber",
    "clothing_fiber": "fiber",
    "cup_ceramic": "ceramic",
    "tube_cement": "cement",
    "brick_clay": "cement",
    "cardboard_paper": "paper",
    "snack_wrapper_paper": "paper",
    "branch_wood": "wood",
    "furniture_wood": "wood",
    "plant": "wood",
    "rov_cable": "fiber",
    "unknown_instance": "unknown",
}

# ============================================================================
# SPLIT CONFIGURATION
# ============================================================================

TRAIN_DIVES = {
    "Bistrina": ["Paralenz Vaquita Gen 2", "SIP-E323CV"],
    "Slano": ["Bluerobotics HD"],
    "Lokrum": ["Bluerobotics HD", "Paralenz Vaquita Gen 2"],
    "Jakljan": ["Bluerobotics HD"],
    "Marseille": ["SIP-E323CV"],
}

TEST_DIVES = {
    "Bistrina": ["Bluerobotics HD"],
    "Slano": ["Paralenz Vaquita"],
    "Lokrum": ["SIP-E323CV"],
    "Jakljan": ["Paralenz Vaquita"],
}


def build_image_lookup(root: Path) -> Dict[str, tuple]:
    """
    Build a lookup table mapping image filenames to (site, dive) tuples.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    image_lookup = {}

    for site in os.listdir(root):
        site_path = root / site
        if not site_path.is_dir():
            continue
        for dive in os.listdir(site_path):
            dive_path = site_path / dive
            if not dive_path.is_dir():
                continue
            for file in os.listdir(dive_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_lookup[file] = (site, dive)

    return image_lookup


def get_split_image_ids(
    coco: Dict[str, Any],
    image_lookup: Dict[str, tuple],
    dive_config: Dict[str, List[str]],
) -> Set[int]:
    """
    Get image IDs that belong to the specified dives.
    """
    image_ids = set()

    for img in coco["images"]:
        file_name = img["file_name"]
        if file_name not in image_lookup:
            continue
        site, dive = image_lookup[file_name]
        if site in dive_config and dive in dive_config[site]:
            image_ids.add(img["id"])

    return image_ids


def create_mapped_categories(
    original_categories: List[Dict],
    class_mapping: Dict[str, str],
) -> tuple:
    """
    Create new category list and ID mapping based on class mapping.
    Returns (new_categories, old_id_to_new_id mapping).
    """
    # Build original id -> name mapping
    orig_id_to_name = {cat["id"]: cat["name"] for cat in original_categories}

    # Get unique mapped class names
    mapped_names = sorted(set(class_mapping.values()))

    # Create new categories with sequential IDs starting from 1
    new_categories = []
    name_to_new_id = {}
    for idx, name in enumerate(mapped_names, start=1):
        new_categories.append({
            "id": idx,
            "name": name,
            "supercategory": name,
        })
        name_to_new_id[name] = idx

    # Create old_id -> new_id mapping
    old_id_to_new_id = {}
    for old_id, old_name in orig_id_to_name.items():
        if old_name in class_mapping:
            new_name = class_mapping[old_name]
            old_id_to_new_id[old_id] = name_to_new_id[new_name]

    return new_categories, old_id_to_new_id


def create_split_coco(
    coco: Dict[str, Any],
    image_ids: Set[int],
    image_lookup: Dict[str, tuple],
    categories: List[Dict],
    category_id_mapping: Dict[int, int] = None,
) -> Dict[str, Any]:
    """
    Create a COCO format dict for the specified image IDs.
    Image file_names are formatted as 'site/dive/imgname'.
    """
    # Filter and transform images
    new_images = []
    new_image_id_map = {}  # old_id -> new_id (sequential)

    for new_idx, img in enumerate(coco["images"]):
        if img["id"] not in image_ids:
            continue

        file_name = img["file_name"]
        site, dive = image_lookup[file_name]

        new_img = img.copy()
        new_img["id"] = new_idx
        new_img["file_name"] = f"{site}/{dive}/{file_name}"
        new_images.append(new_img)
        new_image_id_map[img["id"]] = new_idx

    # Filter and transform annotations
    new_annotations = []
    ann_idx = 0
    for ann in coco["annotations"]:
        if ann["image_id"] not in image_ids:
            continue

        new_ann = ann.copy()
        new_ann["id"] = ann_idx
        new_ann["image_id"] = new_image_id_map[ann["image_id"]]

        # Map category ID if mapping provided
        if category_id_mapping is not None:
            old_cat_id = ann["category_id"]
            if old_cat_id in category_id_mapping:
                new_ann["category_id"] = category_id_mapping[old_cat_id]
            else:
                continue  # Skip annotations with unmapped categories

        new_annotations.append(new_ann)
        ann_idx += 1

    return {
        "categories": categories,
        "images": new_images,
        "annotations": new_annotations,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SeaClear Marine Debris Dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to SeaClear dataset directory (containing dataset.json)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for split files (default: dataset_path/splits)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path / "splits"

    # Verify required files exist
    dataset_json = dataset_path / "dataset.json"
    if not dataset_json.exists():
        print(f"Error: dataset.json not found at: {dataset_json}")
        sys.exit(1)

    # Verify dataset directory structure
    if not dataset_path.is_dir():
        print(f"Error: Dataset path is not a directory: {dataset_path}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Preparing SeaClear Marine Debris Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Load dataset
    print("Loading dataset...")
    with open(dataset_json) as f:
        coco = json.load(f)

    print(f"  Images: {len(coco['images'])}")
    print(f"  Annotations: {len(coco['annotations'])}")
    print(f"  Categories: {len(coco['categories'])}")
    print()

    # Build image lookup
    print("Building image lookup...")
    image_lookup = build_image_lookup(dataset_path)
    print(f"  Found {len(image_lookup)} images on disk")
    print()

    # Get train/test image IDs
    print("Splitting by dives...")
    train_ids = get_split_image_ids(coco, image_lookup, TRAIN_DIVES)
    test_ids = get_split_image_ids(coco, image_lookup, TEST_DIVES)

    print(f"  Train images: {len(train_ids)}")
    print(f"  Test images: {len(test_ids)}")
    print()

    # Print split configuration
    print("Train dives:")
    for site, dives in TRAIN_DIVES.items():
        print(f"  {site}: {', '.join(dives)}")
    print()
    print("Test dives:")
    for site, dives in TEST_DIVES.items():
        print(f"  {site}: {', '.join(dives)}")
    print()

    # Create splits for each class configuration
    splits = [
        ("base", coco["categories"], None),
        ("material", *create_mapped_categories(coco["categories"], CLASS_TO_MATERIAL)),
        ("superclass", *create_mapped_categories(coco["categories"], CLASS_TO_SUPERCLASS)),
    ]

    print("-" * 60)
    print("Creating split files...")
    print("-" * 60)

    for split_name, categories, cat_id_mapping in splits:
        # Create train split
        train_coco = create_split_coco(
            coco, train_ids, image_lookup, categories, cat_id_mapping
        )
        train_path = output_dir / f"{split_name}_train.json"
        with open(train_path, "w") as f:
            json.dump(train_coco, f)
        print(f"  {split_name}_train.json: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")

        # Create test split
        test_coco = create_split_coco(
            coco, test_ids, image_lookup, categories, cat_id_mapping
        )
        test_path = output_dir / f"{split_name}_test.json"
        with open(test_path, "w") as f:
            json.dump(test_coco, f)
        print(f"  {split_name}_test.json: {len(test_coco['images'])} images, {len(test_coco['annotations'])} annotations")

    print()
    print("=" * 60)
    print("SeaClear dataset preparation complete!")
    print("=" * 60)
    print()
    print("Created split files:")
    print(f"  - base_train.json, base_test.json (40 classes)")
    print(f"  - material_train.json, material_test.json (11 classes)")
    print(f"  - superclass_train.json, superclass_test.json (5 classes)")
    print()
    print("Image file names are formatted as: site/dive/filename.jpg")
    print(f"  Example: Bistrina/Bluerobotics HD/1.jpg")


if __name__ == "__main__":
    main()
