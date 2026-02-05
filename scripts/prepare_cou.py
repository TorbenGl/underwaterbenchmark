#!/usr/bin/env python3
"""
Prepare COU (Coral Object Understanding) Dataset with Metal Rod class removed.

This script processes the COU dataset COCO annotations to:
1. Remove the "Metal Rod" class (original category_id 4)
2. Reduce all subsequent class IDs by 1 (5→4, 6→5, ..., 24→23)
3. Filter out Metal Rod annotations and print warnings

Input dataset structure expected:
    {DATA_ROOT}/cou/coco/coco/
        images/
            *.jpg / *.png
        train_annotations.json
        val_annotations.json
        test_annotations.json (optional)

Output:
    Creates new annotation files with '_no_metalrod' suffix:
        train_annotations_no_metalrod.json
        val_annotations_no_metalrod.json
        test_annotations_no_metalrod.json (if input exists)

Original COU classes in COCO file (categories start at 0):
    0: Unknown Instance -> 0
    1: Scissors         -> 1
    2: Plastic Cup      -> 2
    3: Metal Rod        <- REMOVED
    4: Fork             -> 3
    5: Bottle           -> 4
    6: Soda Can         -> 5
    7: Case             -> 6
    8: Plastic Bag      -> 7
    9: Cup              -> 8
    10: Goggles         -> 9
    11: Flipper         -> 10
    12: LoCo            -> 11
    13: Aqua            -> 12
    14: Pipe            -> 13
    15: Snorkel         -> 14
    16: Spoon           -> 15
    17: Lure            -> 16
    18: Screwdriver     -> 17
    19: Car             -> 18
    20: Tripod          -> 19
    21: ROV             -> 20
    22: Knife           -> 21
    23: Dive Weight     -> 22

Note: The datamodule preset adds background at 0 via increase_idx=True,
shifting all these indices by +1 in the final model output.

Usage:
    python scripts/prepare_cou.py /path/to/cou/coco/coco
    python scripts/prepare_cou.py /path/to/cou/coco/coco --output-dir /path/to/output
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Metal Rod category ID to remove (in this COCO file, categories start at 0)
METAL_ROD_CATEGORY_ID = 3

# Original COU categories (as found in COCO file, starting at 0)
ORIGINAL_CATEGORIES = {
    0: "Unknown Instance",
    1: "Scissors",
    2: "Plastic Cup",
    3: "Metal Rod",
    4: "Fork",
    5: "Bottle",
    6: "Soda Can",
    7: "Case",
    8: "Plastic Bag",
    9: "Cup",
    10: "Goggles",
    11: "Flipper",
    12: "LoCo",
    13: "Aqua",
    14: "Pipe",
    15: "Snorkel",
    16: "Spoon",
    17: "Lure",
    18: "Screwdriver",
    19: "Car",
    20: "Tripod",
    21: "ROV",
    22: "Knife",
    23: "Dive Weight",
}


def build_category_mapping() -> Tuple[Dict[int, int], List[Dict[str, Any]]]:
    """
    Build the category ID mapping and new category list.

    Returns:
        Tuple of (old_id_to_new_id mapping, new_categories list)
    """
    old_to_new = {}
    new_categories = []

    new_id = 0  # Start at 0 since COCO file categories start at 0
    for old_id in sorted(ORIGINAL_CATEGORIES.keys()):
        if old_id == METAL_ROD_CATEGORY_ID:
            # Skip Metal Rod - mark as None to filter out
            old_to_new[old_id] = None
            continue

        old_to_new[old_id] = new_id
        new_categories.append({
            "id": new_id,
            "name": ORIGINAL_CATEGORIES[old_id],
            "supercategory": "object",
        })
        new_id += 1

    return old_to_new, new_categories


def process_coco_file(
    input_path: Path,
    output_path: Path,
    category_mapping: Dict[int, int],
    new_categories: List[Dict[str, Any]],
) -> Dict[str, int]:
    """
    Process a single COCO annotation file.

    Args:
        input_path: Path to input COCO JSON file
        output_path: Path to output COCO JSON file
        category_mapping: Mapping from old category IDs to new IDs (None = remove)
        new_categories: List of new category definitions

    Returns:
        Dict with statistics (images_total, annotations_total, annotations_removed, etc.)
    """
    stats = {
        "images_total": 0,
        "annotations_total": 0,
        "annotations_removed": 0,
        "images_with_removed_annotations": set(),
    }

    # Load the COCO file
    with open(input_path, "r") as f:
        coco = json.load(f)

    stats["images_total"] = len(coco.get("images", []))
    stats["annotations_total"] = len(coco.get("annotations", []))

    # Build image ID to filename mapping for warning messages
    image_id_to_filename = {
        img["id"]: img.get("file_name", f"image_id_{img['id']}")
        for img in coco.get("images", [])
    }

    # Process annotations
    new_annotations = []
    new_ann_id = 0

    for ann in coco.get("annotations", []):
        old_cat_id = ann["category_id"]
        new_cat_id = category_mapping.get(old_cat_id)

        if new_cat_id is None:
            # This is a Metal Rod annotation - remove it
            stats["annotations_removed"] += 1
            image_id = ann["image_id"]
            image_filename = image_id_to_filename.get(image_id, f"image_id_{image_id}")
            stats["images_with_removed_annotations"].add(image_filename)

            # Print warning
            print(f"  WARNING: Removed Metal Rod annotation (ann_id={ann['id']}) "
                  f"from image: {image_filename}")
            continue

        # Remap the category ID
        new_ann = ann.copy()
        new_ann["id"] = new_ann_id
        new_ann["category_id"] = new_cat_id
        new_annotations.append(new_ann)
        new_ann_id += 1

    # Build new COCO structure
    new_coco = {
        "images": coco.get("images", []),
        "annotations": new_annotations,
        "categories": new_categories,
    }

    # Preserve any additional top-level keys (info, licenses, etc.)
    for key in coco:
        if key not in ["images", "annotations", "categories"]:
            new_coco[key] = coco[key]

    # Write output
    with open(output_path, "w") as f:
        json.dump(new_coco, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare COU dataset with Metal Rod class removed"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to COU dataset directory (containing annotation JSON files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for processed files (default: same as input)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path

    # Verify dataset path exists
    if not dataset_path.is_dir():
        print(f"Error: Dataset path is not a directory: {dataset_path}")
        sys.exit(1)

    # Define annotation files to process
    annotation_files = [
        "train_annotations.json",
        "val_annotations.json",
        "test_annotations.json",
    ]

    # Check which annotation files exist
    existing_files = []
    for ann_file in annotation_files:
        if (dataset_path / ann_file).exists():
            existing_files.append(ann_file)

    if not existing_files:
        print(f"Error: No annotation files found in: {dataset_path}")
        print("Expected files: train_annotations.json, val_annotations.json, test_annotations.json")
        sys.exit(1)

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the category mapping
    category_mapping, new_categories = build_category_mapping()

    print("=" * 70)
    print("Preparing COU Dataset - Removing Metal Rod Class")
    print("=" * 70)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    print("Category Mapping (old -> new):")
    print("-" * 40)
    for old_id in sorted(ORIGINAL_CATEGORIES.keys()):
        old_name = ORIGINAL_CATEGORIES[old_id]
        new_id = category_mapping[old_id]
        if new_id is None:
            print(f"  {old_id:2d}: {old_name:<20} -> REMOVED")
        else:
            print(f"  {old_id:2d}: {old_name:<20} -> {new_id}")
    print()

    print(f"New category count: {len(new_categories)} (was {len(ORIGINAL_CATEGORIES)})")
    print()

    # Process each annotation file
    total_stats = {
        "files_processed": 0,
        "total_annotations_removed": 0,
        "total_images_affected": set(),
    }

    print("-" * 70)
    print("Processing annotation files...")
    print("-" * 70)

    for ann_file in existing_files:
        input_path = dataset_path / ann_file
        output_filename = ann_file.replace(".json", "_no_metalrod.json")
        output_path = output_dir / output_filename

        print()
        print(f"Processing: {ann_file}")
        print(f"  Output: {output_filename}")

        stats = process_coco_file(
            input_path, output_path, category_mapping, new_categories
        )

        total_stats["files_processed"] += 1
        total_stats["total_annotations_removed"] += stats["annotations_removed"]
        total_stats["total_images_affected"].update(stats["images_with_removed_annotations"])

        print()
        print(f"  Images: {stats['images_total']}")
        print(f"  Annotations (original): {stats['annotations_total']}")
        print(f"  Annotations removed: {stats['annotations_removed']}")
        print(f"  Annotations (final): {stats['annotations_total'] - stats['annotations_removed']}")
        print(f"  Images with removed annotations: {len(stats['images_with_removed_annotations'])}")

    # Print summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Files processed: {total_stats['files_processed']}")
    print(f"Total Metal Rod annotations removed: {total_stats['total_annotations_removed']}")
    print(f"Total images affected: {len(total_stats['total_images_affected'])}")
    print()

    if total_stats["total_annotations_removed"] > 0:
        print("WARNING: Metal Rod annotations were found and removed!")
        print("The following images had Metal Rod annotations removed:")
        for img in sorted(total_stats["total_images_affected"]):
            print(f"  - {img}")
    else:
        print("No Metal Rod annotations found in the dataset.")

    print()
    print("=" * 70)
    print("COU dataset preparation complete!")
    print("=" * 70)
    print()
    print("Output files created:")
    for ann_file in existing_files:
        output_filename = ann_file.replace(".json", "_no_metalrod.json")
        print(f"  - {output_filename}")
    print()
    print("New class mapping in COCO file:")
    for cat in new_categories:
        print(f"  {cat['id']}: {cat['name']}")
    print()
    print("Note: With increase_idx=True and fill_background=True in the datamodule,")
    print("the final model will have background at 0, and these classes shifted by +1.")


if __name__ == "__main__":
    main()
