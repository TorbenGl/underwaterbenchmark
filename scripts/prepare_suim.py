#!/usr/bin/env python3
"""
Prepare SUIM (Semantic Segmentation of Underwater Imagery) dataset for training.

This script creates train/val/test splits from the SUIM dataset structure.

SUIM structure:
    suim/
        train_val/
            images/*.jpg
            masks/*.bmp
        TEST/
            images/*.jpg
            masks/*.bmp

Output format (ImageMaskPreset):
    {
        "samples": [
            {"image": "train_val/images/file.jpg", "mask": "train_val/masks/file.bmp"},
            ...
        ]
    }

Usage:
    python scripts/prepare_suim.py /path/to/suim

This creates:
    - train_samples.json
    - val_samples.json
    - test_samples.json

SUIM Classes (8 classes, RGB color-coded masks):
    0: Background waterbody (BW) - RGB(0, 0, 0)
    1: Human divers (HD) - RGB(0, 0, 255)
    2: Plants/sea-grass (PF) - RGB(0, 255, 0)
    3: Wrecks/ruins (WR) - RGB(0, 255, 255)
    4: Robots/instruments (RO) - RGB(255, 0, 0)
    5: Reefs and invertebrates (RI) - RGB(255, 0, 255)
    6: Fish and vertebrates (FV) - RGB(255, 255, 0)
    7: Sand/sea-floor (SR) - RGB(255, 255, 255)
"""

import argparse
import json
import random
import sys
from pathlib import Path


def find_image_mask_pairs(images_dir: Path, masks_dir: Path, prefix: str) -> list:
    """
    Find matching image/mask pairs.

    Args:
        images_dir: Directory containing images
        masks_dir: Directory containing masks
        prefix: Path prefix for relative paths (e.g., "train_val")

    Returns:
        List of {"image": ..., "mask": ...} dictionaries
    """
    samples = []

    # Get all images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))

    for img_path in sorted(image_files):
        # Find corresponding mask (same name, .bmp extension)
        mask_name = img_path.stem + ".bmp"
        mask_path = masks_dir / mask_name

        if mask_path.exists():
            samples.append({
                "image": f"{prefix}/images/{img_path.name}",
                "mask": f"{prefix}/masks/{mask_name}",
            })
        else:
            print(f"Warning: No mask found for {img_path.name}")

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SUIM dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to SUIM dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for annotation files (default: same as dataset_path)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Fraction of train_val to use for validation (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path

    # Verify directories exist
    train_val_dir = dataset_path / "train_val"
    test_dir = dataset_path / "TEST"

    if not train_val_dir.exists():
        print(f"Error: train_val directory not found: {train_val_dir}")
        sys.exit(1)
    if not test_dir.exists():
        print(f"Error: TEST directory not found: {test_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Preparing SUIM Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print(f"Validation split: {args.val_split * 100:.0f}%")
    print(f"Random seed: {args.seed}")
    print()

    # Find all train_val samples
    train_val_samples = find_image_mask_pairs(
        train_val_dir / "images",
        train_val_dir / "masks",
        "train_val"
    )
    print(f"Found {len(train_val_samples)} samples in train_val/")

    # Find all test samples
    test_samples = find_image_mask_pairs(
        test_dir / "images",
        test_dir / "masks",
        "TEST"
    )
    print(f"Found {len(test_samples)} samples in TEST/")

    # Split train_val into train and val
    random.seed(args.seed)
    random.shuffle(train_val_samples)

    num_val = int(len(train_val_samples) * args.val_split)
    val_samples = train_val_samples[:num_val]
    train_samples = train_val_samples[num_val:]

    print()
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Write annotation files
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, samples in [
        ("train_samples.json", train_samples),
        ("val_samples.json", val_samples),
        ("test_samples.json", test_samples),
    ]:
        output_file = output_dir / name
        with open(output_file, "w") as f:
            json.dump({"samples": samples}, f, indent=2)
        print(f"Created: {output_file}")

    print()
    print("=" * 60)
    print("SUIM dataset preparation complete!")
    print("=" * 60)
    print()
    print("Created annotation files:")
    print(f"  - train_samples.json ({len(train_samples)} samples)")
    print(f"  - val_samples.json ({len(val_samples)} samples)")
    print(f"  - test_samples.json ({len(test_samples)} samples)")
    print()
    print("You can now use the 'suim' dataset preset:")
    print("  python train_semantic.py --dataset suim --batch_size 8")


if __name__ == "__main__":
    main()
