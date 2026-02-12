#!/usr/bin/env python3
"""
Prepare DeepFish Segmentation dataset for training.

This script converts the DeepFish CSV split files to the ImageMaskPreset format.

DeepFish Segmentation structure:
    Segmentation/
        images/
            valid/*.jpg    (frames with fish)
            empty/*.jpg    (frames without fish)
        masks/
            valid/*.png    (grayscale: 0=background, 255=fish)
            empty/*.png    (all background)
        train.csv
        val.csv
        test.csv

Output format (ImageMaskPreset):
    {
        "samples": [
            {"image": "images/valid/X.jpg", "mask": "masks/valid/X.png"},
            ...
        ]
    }

Usage:
    python scripts/prepare_deepfish.py /path/to/DeepFish/Segmentation

This creates:
    - train_samples.json
    - val_samples.json
    - test_samples.json

DeepFish Segmentation Classes (binary):
    0: Background - grayscale value 0
    1: Fish       - grayscale value 255
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def convert_csv_to_samples(csv_file: Path, dataset_path: Path) -> list:
    """
    Read a DeepFish CSV split file and produce image/mask sample pairs.

    Args:
        csv_file: Path to CSV file (train.csv, val.csv, test.csv)
        dataset_path: Root path of the Segmentation dataset

    Returns:
        List of {"image": ..., "mask": ...} dictionaries
    """
    samples = []
    missing = []

    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_id = row["ID"]  # e.g. "valid/7623_F2_f000460"
            image_rel = f"images/{sample_id}.jpg"
            mask_rel = f"masks/{sample_id}.png"

            image_path = dataset_path / image_rel
            mask_path = dataset_path / mask_rel

            if not image_path.exists() or not mask_path.exists():
                missing.append(sample_id)
                continue

            samples.append({
                "image": image_rel,
                "mask": mask_rel,
            })

    if missing:
        print(f"  Warning: {len(missing)} samples missing on disk:")
        for m in missing[:5]:
            print(f"    - {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare DeepFish Segmentation dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to DeepFish Segmentation directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for annotation files (default: same as dataset_path)",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path

    # Verify required CSV files exist
    splits = {
        "train": dataset_path / "train.csv",
        "val": dataset_path / "val.csv",
        "test": dataset_path / "test.csv",
    }

    missing = [str(f) for name, f in splits.items() if not f.exists()]
    if missing:
        print("Error: Missing required CSV files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Verify image/mask directories exist
    for subdir in ["images/valid", "images/empty", "masks/valid", "masks/empty"]:
        d = dataset_path / subdir
        if not d.exists():
            print(f"Error: Directory not found: {d}")
            sys.exit(1)

    print("=" * 60)
    print("Preparing DeepFish Segmentation Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Convert each split
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, csv_file in splits.items():
        print(f"Processing {split_name} split ({csv_file.name})...")
        samples = convert_csv_to_samples(csv_file, dataset_path)

        output_file = output_dir / f"{split_name}_samples.json"
        with open(output_file, "w") as f:
            json.dump({"samples": samples}, f, indent=2)

        # Count valid vs empty
        valid_count = sum(1 for s in samples if s["image"].startswith("images/valid"))
        empty_count = sum(1 for s in samples if s["image"].startswith("images/empty"))
        print(f"  -> {output_file.name}: {len(samples)} samples ({valid_count} valid, {empty_count} empty)")

    print()
    print("=" * 60)
    print("DeepFish dataset preparation complete!")
    print("=" * 60)
    print()
    print("Created annotation files:")
    for split_name in splits:
        print(f"  - {split_name}_samples.json")
    print()
    print("You can now use the 'deepfish' dataset preset:")
    print("  python train_semantic.py --dataset deepfish --batch_size 8")


if __name__ == "__main__":
    main()
