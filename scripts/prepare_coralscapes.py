#!/usr/bin/env python3
"""
Prepare Coralscapes dataset for training with remapped class indices.

Remaps the background and dark classes to index 0 (merged), shifting other classes accordingly:
- Original: background=13, dark=14, classes go from 1-39
- Remapped: background + dark → 0, classes 1-12 stay same, classes 15-39 shift down to 13-37
- Result: 38 total classes (merged background+dark into one)

This script processes all mask values in parquet files (HuggingFace dataset format) and saves
remapped versions, so the dataloader doesn't need to remap on every batch.

Supports both formats:
1. Parquet files (HuggingFace datasets):
    coralscapes/data/
        train-*.parquet
        validation-*.parquet
        test-*.parquet

2. Local image/mask files:
    coralscapes/
        train/image/ and train/label/ subdirectories
        validation/image/ and validation/label/ subdirectories
        test/image/ and test/label/ subdirectories

Usage:
    python scripts/prepare_coralscapes.py /path/to/coralscapes

This remaps mask values in-place in the dataset directory.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import pandas as pd
import io
import json


def update_id2label_mapping(dataset_path: Path) -> bool:
    """
    Update id2label.json file to reflect the remapped class indices.
    
    Args:
        dataset_path: Path to coralscapes dataset
        
    Returns:
        True if file was updated, False if not found
    """
    id2label_path = dataset_path / "id2label.json"
    
    if not id2label_path.exists():
        return False
    
    print("Updating id2label.json mapping...")
    
    # Define remapped ID to label mapping
    # Merges background (13) and dark (14) into class 0
    # Classes 15-39 shift down by 2
    remapped_mapping = {
        "0": "background",  # Merged: original background (13) + dark (14)
        "1": "seagrass",
        "2": "trash",
        "3": "other coral dead",
        "4": "other coral bleached",
        "5": "sand",
        "6": "other coral alive",
        "7": "human",
        "8": "transect tools",
        "9": "fish",
        "10": "algae covered substrate",
        "11": "other animal",
        "12": "unknown hard substrate",
        "13": "transect line",  # Was 15
        "14": "massive/meandering bleached",  # Was 16
        "15": "massive/meandering alive",  # Was 17
        "16": "rubble",  # Was 18
        "17": "branching bleached",  # Was 19
        "18": "branching dead",  # Was 20
        "19": "millepora",  # Was 21
        "20": "branching alive",  # Was 22
        "21": "massive/meandering dead",  # Was 23
        "22": "clam",  # Was 24
        "23": "acropora alive",  # Was 25
        "24": "sea cucumber",  # Was 26
        "25": "turbinaria",  # Was 27
        "26": "table acropora alive",  # Was 28
        "27": "sponge",  # Was 29
        "28": "anemone",  # Was 30
        "29": "pocillopora alive",  # Was 31
        "30": "table acropora dead",  # Was 32
        "31": "meandering bleached",  # Was 33
        "32": "stylophora alive",  # Was 34
        "33": "sea urchin",  # Was 35
        "34": "meandering alive",  # Was 36
        "35": "meandering dead",  # Was 37
        "36": "crown of thorn",  # Was 38
        "37": "dead clam",  # Was 39
    }
    
    # Write remapped mapping
    with open(id2label_path, "w") as f:
        json.dump(remapped_mapping, f)
    
    print(f"  Updated {id2label_path}")
    return True


def check_if_remapped(dataset_path: Path) -> bool:
    """
    Check if dataset has already been remapped by looking for class 0 (which we insert).

    Original data: class 0 does not exist (background is at 13)
    Remapped data: class 0 exists (remapped background)

    Args:
        dataset_path: Path to coralscapes dataset

    Returns:
        True if class 0 found (already remapped), False otherwise
    """
    # Check for parquet format first (HuggingFace datasets)
    data_dir = dataset_path / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.glob("*.parquet"))
        if parquet_files:
            # Check first parquet file
            df = pd.read_parquet(parquet_files[0])
            if "label" in df.columns:
                # Get first non-null label
                for label_obj in df["label"]:
                    if label_obj is not None:
                        label_array = np.array(label_obj)
                        if np.any(label_array == 0):
                            return True
                        break
            return False
    
    # Check for local image/mask format
    splits = ["train", "validation", "test"]
    
    for split in splits:
        split_path = dataset_path / split
        label_dir = split_path / "label"
        
        if not label_dir.exists():
            continue
        
        mask_files = list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))
        
        if not mask_files:
            continue
        
        # Check first mask for class 0
        mask = np.array(Image.open(mask_files[0]))
        if np.any(mask == 0):
            return True
    
    return False


def remap_mask(mask_array: np.ndarray) -> np.ndarray:
    """
    Remap mask values: 
    - background (13) + dark (14) → 0 (merged)
    - classes 1-12 stay the same
    - classes 15-39 shift down by 2
    
    Args:
        mask_array: Original mask with background=13, dark=14
        
    Returns:
        Remapped mask with background+dark=0
    """
    remapped = mask_array.copy()
    
    # Map background (13) and dark (14) both to 0 (merge)
    remapped[mask_array == 13] = 0
    remapped[mask_array == 14] = 0
    
    # Classes 1-12 stay the same (already in correct order)
    # No action needed for classes 1-12
    
    # Shift classes 15-39 down by 2 (to fill the gap where 13-14 were)
    for class_id in range(15, 40):
        remapped[mask_array == class_id] = class_id - 2
    
    return remapped


def process_parquet_files(dataset_path: Path) -> int:
    """
    Process parquet files (HuggingFace dataset format).
    
    Args:
        dataset_path: Path to coralscapes dataset
        
    Returns:
        Number of samples processed
    """
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        return 0
    
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        return 0
    
    sample_count = 0
    
    for parquet_file in parquet_files:
        print(f"Processing {parquet_file.name}...")
        
        df = pd.read_parquet(parquet_file)
        
        if "label" not in df.columns:
            print(f"  Warning: 'label' column not found, skipping")
            continue
        
        # Remap all labels in this parquet file
        remapped_labels = []
        for label_obj in df["label"]:
            if label_obj is not None:
                # Handle different label formats
                if isinstance(label_obj, Image.Image):
                    # Already a PIL Image - convert to remapped
                    label_array = np.array(label_obj)
                    remapped = remap_mask(label_array)
                    remapped_img = Image.fromarray(remapped.astype(np.uint8))
                    # Convert back to bytes dict format for parquet
                    img_bytes = io.BytesIO()
                    remapped_img.save(img_bytes, format="PNG")
                    remapped_labels.append({"bytes": img_bytes.getvalue()})
                    
                elif isinstance(label_obj, np.ndarray):
                    # Already a numpy array
                    remapped = remap_mask(label_obj)
                    remapped_img = Image.fromarray(remapped.astype(np.uint8))
                    # Convert to bytes dict format for parquet
                    img_bytes = io.BytesIO()
                    remapped_img.save(img_bytes, format="PNG")
                    remapped_labels.append({"bytes": img_bytes.getvalue()})
                    
                elif isinstance(label_obj, dict):
                    # HuggingFace Image feature format: {"path": "...", "bytes": ...}
                    if "bytes" in label_obj:
                        label_array = np.array(Image.open(io.BytesIO(label_obj["bytes"])))
                        remapped = remap_mask(label_array)
                        remapped_img = Image.fromarray(remapped.astype(np.uint8))
                        # Keep dict format but update bytes
                        img_bytes = io.BytesIO()
                        remapped_img.save(img_bytes, format="PNG")
                        remapped_labels.append({"bytes": img_bytes.getvalue()})
                    else:
                        print(f"    Warning: Unknown label dict format, skipping sample")
                        remapped_labels.append(label_obj)
                else:
                    print(f"    Warning: Unknown label type {type(label_obj)}, skipping sample")
                    remapped_labels.append(label_obj)
            else:
                remapped_labels.append(None)
        
        df["label"] = remapped_labels
        
        # Save back to parquet
        df.to_parquet(parquet_file)
        sample_count += len(df)
        print(f"  Processed {len(df)} samples")
    
    return sample_count


def process_image_mask_files(dataset_path: Path) -> int:
    """
    Process local image/mask files.
    
    Args:
        dataset_path: Path to coralscapes dataset
        
    Returns:
        Number of masks processed
    """
    mask_count = 0
    splits = ["train", "validation", "test"]
    
    for split in splits:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"Warning: {split_path} does not exist, skipping")
            continue
        
        label_dir = split_path / "label"
        if not label_dir.exists():
            print(f"Warning: {label_dir} does not exist, skipping {split}")
            continue
        
        # Find all mask files
        mask_files = list(label_dir.glob("*.png")) + list(label_dir.glob("*.jpg"))
        
        if not mask_files:
            print(f"No mask files found in {label_dir}")
            continue
        
        print(f"Processing {split} split ({len(mask_files)} masks)...")
        
        for mask_path in mask_files:
            # Load mask
            mask = np.array(Image.open(mask_path))
            
            # Remap
            remapped = remap_mask(mask)
            
            # Save back
            Image.fromarray(remapped.astype(np.uint8)).save(mask_path)
            mask_count += 1
            
            if mask_count % 100 == 0:
                print(f"  Processed {mask_count} masks...")
    
    return mask_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Coralscapes dataset for training with remapped class indices"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to Coralscapes dataset directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying files"
    )
    
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        sys.exit(1)
    
    # Check if already remapped
    if check_if_remapped(dataset_path):
        print("Error: Dataset appears to be already remapped (class 0 found in masks)")
        print("Run this script only once on the original dataset.")
        sys.exit(1)
    
    if args.dry_run:
        print("[DRY RUN] Would remap classes in:")
        print(f"  {dataset_path}")
        print("\nRemapping scheme:")
        print("  Original class 13 (background) -> New class 0")
        print("  Original class 14 (dark) -> New class 0 (merged with background)")
        print("  Original classes 1-12 -> New classes 1-12 (unchanged)")
        print("  Original classes 15-39 -> New classes 13-37 (shifted down by 2)")
        print("\nResult: 38 total classes (0-37)")
        return

    print(f"Remapping Coralscapes dataset at {dataset_path}")
    print("Background (13) + Dark (14) merged to 0, classes 15-39 shifted down by 2\n")
    
    # Check which format we have
    data_dir = dataset_path / "data"
    if data_dir.exists() and list(data_dir.glob("*.parquet")):
        print("Detected parquet format (HuggingFace dataset)...")
        count = process_parquet_files(dataset_path)
    else:
        print("Detected local image/mask format...")
        count = process_image_mask_files(dataset_path)
    
    # Update id2label.json if it exists
    update_id2label_mapping(dataset_path)
    
    print(f"\nSuccess! Remapped {count} samples/masks.")
    print("Update your preset to use the new class mapping:")
    print("  - background + dark merged to index 0")
    print("  - num_classes = 38 (was 39, merged background+dark)")


if __name__ == "__main__":
    main()
