#!/usr/bin/env python3
"""
Prepare USOD10K (Underwater Salient Object Detection 10K) dataset for training.

This script creates annotation files for the ImageMaskPreset format.

USOD10K structure:
    USOD10k/USOD10k/
        TR/
            RGB/*.png (images)
            GT/*.png (saliency masks)
        VAL/
            RGB/*.png
            GT/*.png
        TE/
            RGB/*.png
            GT/GT/*.png (nested structure)

Output format (ImageMaskPreset):
    {
        "samples": [
            {"image": "TR/RGB/00001.png", "mask": "TR/GT/00001.png"},
            ...
        ]
    }

Usage:
    python scripts/prepare_usod10k.py /path/to/USOD10k/USOD10k

This creates:
    - train_samples.json
    - val_samples.json
    - test_samples.json

Note: USOD10K masks are saliency maps (grayscale 0-255).
For binary segmentation, threshold at 128 or use value_to_class mapping.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def find_image_mask_pairs(split_dir: Path, split_name: str) -> list:
    """
    Find matching image/mask pairs in a split directory.

    Args:
        split_dir: Directory containing RGB/ and GT/ subdirectories
        split_name: Name of the split (TR, VAL, TE)

    Returns:
        List of {"image": ..., "mask": ...} dictionaries
    """
    samples = []

    rgb_dir = split_dir / "RGB"
    gt_dir = split_dir / "GT"

    if not rgb_dir.exists() or not gt_dir.exists():
        return samples

    # Handle nested GT/GT structure (found in TE split)
    if (gt_dir / "GT").exists() and (gt_dir / "GT").is_dir():
        gt_dir = gt_dir / "GT"
        gt_rel_path = f"{split_name}/GT/GT"
    else:
        gt_rel_path = f"{split_name}/GT"

    # Get all RGB images
    rgb_files = sorted(rgb_dir.glob("*.png")) + sorted(rgb_dir.glob("*.jpg"))
    gt_files = set(f.stem for f in gt_dir.glob("*.png"))

    for rgb_path in rgb_files:
        # Check if corresponding GT exists
        if rgb_path.stem in gt_files:
            samples.append({
                "image": f"{split_name}/RGB/{rgb_path.name}",
                "mask": f"{gt_rel_path}/{rgb_path.stem}.png",
            })

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare USOD10K dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to USOD10k/USOD10k directory"
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
    required_dirs = ["TR", "VAL"]
    for d in required_dirs:
        if not (dataset_path / d).exists():
            print(f"Error: {d} directory not found in {dataset_path}")
            sys.exit(1)

    print("=" * 60)
    print("Preparing USOD10K Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Process each split
    splits = {
        "TR": "train_samples.json",
        "VAL": "val_samples.json",
        "TE": "test_samples.json",
    }

    for split_name, output_name in splits.items():
        split_dir = dataset_path / split_name
        if not split_dir.exists():
            print(f"Skipping {split_name} (not found)")
            continue

        samples = find_image_mask_pairs(split_dir, split_name)

        if samples:
            output_file = output_dir / output_name
            with open(output_file, "w") as f:
                json.dump({"samples": samples}, f, indent=2)
            print(f"Created: {output_name} ({len(samples)} samples)")
        else:
            print(f"Skipping {split_name} (no valid pairs found)")

    print()
    print("=" * 60)
    print("USOD10K dataset preparation complete!")
    print("=" * 60)
    print()
    print("Note: USOD10K masks are saliency maps (grayscale 0-255).")
    print("For binary segmentation, the preset uses threshold at 128.")
    print()
    print("You can now use the 'usod10k' dataset preset:")
    print("  python train_semantic.py --dataset usod10k --batch_size 8")


if __name__ == "__main__":
    main()
