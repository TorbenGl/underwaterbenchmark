#!/usr/bin/env python3
"""
Prepare Looking for Seagrass (L4S) dataset for training.

This script converts the L4S JSON format to the ImageMaskPreset format.

L4S format:
    [
        {"image": "images/00d01/file.jpg", "ground-truth": "ground-truth/00d01/pm_file.png"},
        ...
    ]

ImageMaskPreset format:
    {
        "samples": [
            {"image": "images/00d01/file.jpg", "mask": "ground-truth/00d01/pm_file.png"},
            ...
        ]
    }

Usage:
    python scripts/prepare_l4s.py /path/to/l4s/dataset

This creates:
    - train_samples.json
    - val_samples.json
    - test_samples.json
"""

import argparse
import json
import sys
from pathlib import Path


def convert_l4s_json(input_file: Path, output_file: Path) -> int:
    """
    Convert L4S JSON format to ImageMaskPreset format.

    Args:
        input_file: Path to L4S JSON file (train.json, validate.json, test.json)
        output_file: Path to output JSON file

    Returns:
        Number of samples converted
    """
    with open(input_file, "r") as f:
        l4s_data = json.load(f)

    # Convert format
    samples = []
    for item in l4s_data:
        samples.append({
            "image": item["image"],
            "mask": item["ground-truth"],
        })

    # Write in ImageMaskPreset format
    output_data = {"samples": samples}
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    return len(samples)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare L4S (Looking for Seagrass) dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to L4S dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for converted files (default: same as dataset_path)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path

    # Verify required files exist
    required_files = {
        "train": dataset_path / "train.json",
        "validate": dataset_path / "validate.json",
        "test": dataset_path / "test.json",
    }

    missing = [str(f) for name, f in required_files.items() if not f.exists()]
    if missing:
        print("Error: Missing required files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # Verify directories exist
    images_dir = dataset_path / "images"
    gt_dir = dataset_path / "ground-truth"

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"Error: Ground-truth directory not found: {gt_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Preparing L4S (Looking for Seagrass) Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Convert each split
    conversions = [
        ("train.json", "train_samples.json"),
        ("validate.json", "val_samples.json"),
        ("test.json", "test_samples.json"),
    ]

    for input_name, output_name in conversions:
        input_file = dataset_path / input_name
        output_file = output_dir / output_name

        num_samples = convert_l4s_json(input_file, output_file)
        print(f"Converted {input_name} -> {output_name} ({num_samples} samples)")

    print()
    print("=" * 60)
    print("L4S dataset preparation complete!")
    print("=" * 60)
    print()
    print("Created annotation files:")
    print(f"  - train_samples.json")
    print(f"  - val_samples.json")
    print(f"  - test_samples.json")
    print()
    print("You can now use the 'l4s' dataset preset:")
    print("  python train_semantic.py --dataset l4s --batch_size 8")
    print()
    print("Or set the data root if your data is in a different location:")
    print(f"  python train_semantic.py --dataset l4s --data_root {dataset_path}")


if __name__ == "__main__":
    main()
