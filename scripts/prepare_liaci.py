#!/usr/bin/env python3
"""
Prepare LIACi dataset for training.

This script:
1. Splits the COCO annotations based on train_test_split.csv
2. Creates train/test annotation files ready for the LIACi preset

Usage:
    python scripts/prepare_liaci.py /path/to/LIACi_dataset_pretty

The script will create:
    - coco-labels_train.json
    - coco-labels_test.json
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from segdatasets.coco_splitter import split_coco_from_csv


def main():
    parser = argparse.ArgumentParser(
        description="Prepare LIACi dataset for training"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to LIACi_dataset_pretty directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for split files (default: same as dataset_path)"
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_path

    # Verify required files exist
    coco_file = dataset_path / "coco-labels.json"
    csv_file = dataset_path / "train_test_split.csv"
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks" / "segmentation"

    missing = []
    if not coco_file.exists():
        missing.append(str(coco_file))
    if not csv_file.exists():
        missing.append(str(csv_file))
    if not images_dir.exists():
        missing.append(str(images_dir))
    if not masks_dir.exists():
        missing.append(str(masks_dir))

    if missing:
        print("Error: Missing required files/directories:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    print("=" * 60)
    print("Preparing LIACi Dataset")
    print("=" * 60)
    print(f"Dataset path: {dataset_path}")
    print(f"Output directory: {output_dir}")
    print()

    # Split COCO annotations based on CSV
    print("Splitting COCO annotations based on train_test_split.csv...")
    output_files = split_coco_from_csv(
        coco_file=coco_file,
        csv_file=csv_file,
        output_dir=output_dir,
        file_name_column="file_name",
        split_column="split",
    )

    print()
    print("=" * 60)
    print("LIACi dataset preparation complete!")
    print("=" * 60)
    print()
    print("Created annotation files:")
    for split_name, path in output_files.items():
        print(f"  - {split_name}: {path}")
    print()
    print("You can now use the 'liaci' dataset preset:")
    print("  python train_semantic.py --dataset liaci --batch_size 8")
    print()
    print("Or set the data root if your data is in a different location:")
    print(f"  python train_semantic.py --dataset liaci --data_root {dataset_path}")


if __name__ == "__main__":
    main()
