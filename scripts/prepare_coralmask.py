#!/usr/bin/env python3
"""
Prepare CoralMask dataset for training.

This script converts per-image JSON annotations to COCO format.

CoralMask structure:
    CoralMask/
        train/
            images/*.jpg
            jsons/*.json (per-image annotations)
        test/
            images/*.jpg
            jsons/*.json

Per-image JSON format:
    {
        "image": {"file_name": "...", "height": ..., "width": ..., "id": ...},
        "annotations": [
            {"id": ..., "image_id": ..., "area": ..., "segmentation": [...], "bbox": [...]},
            ...
        ]
    }

Output: COCO-format annotation files with a single "coral" category.

Usage:
    python scripts/prepare_coralmask.py /path/to/CoralMask

This creates:
    - train_annotations.json
    - test_annotations.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm


def convert_to_coco(json_dir: Path, images_dir: Path) -> dict:
    """
    Convert per-image JSON files to COCO format.

    Args:
        json_dir: Directory containing per-image JSON files
        images_dir: Directory containing images

    Returns:
        COCO-format annotation dictionary
    """
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "coral", "supercategory": "coral"}],
    }

    json_files = sorted(json_dir.glob("*.json"))
    annotation_id = 0

    for json_file in tqdm(json_files, desc="Processing"):
        with open(json_file) as f:
            data = json.load(f)

        # Get image info
        img_info = data["image"]
        image_id = len(coco_data["images"])

        # Update image info with correct path
        coco_data["images"].append({
            "id": image_id,
            "file_name": img_info["file_name"],
            "height": img_info["height"],
            "width": img_info["width"],
        })

        # Convert annotations
        for ann in data.get("annotations", []):
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # All corals are category 1
                "segmentation": ann.get("segmentation", []),
                "area": ann.get("area", 0),
                "bbox": ann.get("bbox", []),
                "iscrowd": 0,
            })
            annotation_id += 1

    return coco_data


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CoralMask dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to CoralMask directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for annotation files (default: same as dataset_path)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path

    # Verify directories exist
    train_jsons = dataset_path / "train" / "jsons"
    test_jsons = dataset_path / "test" / "jsons"

    if not train_jsons.exists():
        print(f"Error: train/jsons directory not found in {dataset_path}")
        sys.exit(1)

    print("=" * 60)
    print("Preparing CoralMask Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Process train set
    print("Processing train set...")
    train_coco = convert_to_coco(
        train_jsons,
        dataset_path / "train" / "images"
    )
    train_output = output_dir / "train_annotations.json"
    with open(train_output, "w") as f:
        json.dump(train_coco, f)
    print(f"Created: train_annotations.json")
    print(f"  Images: {len(train_coco['images'])}")
    print(f"  Annotations: {len(train_coco['annotations'])}")
    print()

    # Process test set
    if test_jsons.exists():
        print("Processing test set...")
        test_coco = convert_to_coco(
            test_jsons,
            dataset_path / "test" / "images"
        )
        test_output = output_dir / "test_annotations.json"
        with open(test_output, "w") as f:
            json.dump(test_coco, f)
        print(f"Created: test_annotations.json")
        print(f"  Images: {len(test_coco['images'])}")
        print(f"  Annotations: {len(test_coco['annotations'])}")
    else:
        print("Skipping test set (not found)")

    print()
    print("=" * 60)
    print("CoralMask dataset preparation complete!")
    print("=" * 60)
    print()
    print("You can now use the 'coralmask' dataset preset:")
    print("  python train_semantic.py --dataset coralmask --batch_size 8")


if __name__ == "__main__":
    main()
