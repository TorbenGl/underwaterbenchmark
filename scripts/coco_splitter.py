import csv
import json
from pathlib import Path
from typing import Optional
import random
from collections import defaultdict


def split_coco_from_csv(
    coco_file: str | Path,
    csv_file: str | Path,
    output_dir: str | Path,
    file_name_column: str = "file_name",
    split_column: str = "split",
) -> dict[str, Path]:
    """
    Split a COCO annotation file based on a CSV file with predefined splits.

    Args:
        coco_file: Path to the source COCO annotation JSON file.
        csv_file: Path to CSV file with columns for file_name and split.
                  Expected format:
                  ,file_name,split
                  0,image_0001.jpg,train
                  1,image_0002.jpg,test
                  ...
        output_dir: Directory to save the split annotation files.
        file_name_column: Column name for the image file name (default: "file_name").
        split_column: Column name for the split assignment (default: "split").

    Returns:
        Dictionary mapping split names to output file paths.
    """
    coco_file = Path(coco_file)
    csv_file = Path(csv_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CSV and build file_name -> split mapping
    file_to_split: dict[str, str] = {}
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row[file_name_column].strip()
            split = row[split_column].strip()
            file_to_split[file_name] = split

    # Load source COCO file
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    info = coco_data.get("info", {})
    licenses = coco_data.get("licenses", [])

    # Build image lookups
    image_id_to_img = {img["id"]: img for img in images}
    filename_to_image_id = {img["file_name"]: img["id"] for img in images}

    # Build annotation lookup by image_id
    image_id_to_anns = defaultdict(list)
    for ann in annotations:
        image_id_to_anns[ann["image_id"]].append(ann)

    # Group image IDs by split
    splits: dict[str, set[int]] = defaultdict(set)
    for file_name, split_name in file_to_split.items():
        if file_name in filename_to_image_id:
            image_id = filename_to_image_id[file_name]
            splits[split_name].add(image_id)
        else:
            print(f"Warning: {file_name} not found in COCO file, skipping")

    # Generate split files
    output_files = {}
    base_name = coco_file.stem

    for split_name, split_image_ids in splits.items():
        if not split_image_ids:
            continue

        split_images = [image_id_to_img[img_id] for img_id in split_image_ids]
        split_annotations = []
        for img_id in split_image_ids:
            split_annotations.extend(image_id_to_anns[img_id])

        split_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": split_images,
            "annotations": split_annotations,
        }

        output_path = output_dir / f"{base_name}_{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)

        output_files[split_name] = output_path
        print(
            f"{split_name}: {len(split_images)} images, "
            f"{len(split_annotations)} annotations -> {output_path}"
        )

    return output_files


def split_coco_dataset(
    coco_file: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    unused_ratio: float = 0.0,
    seed: Optional[int] = 42,
    stratify_by_category: bool = False,
) -> dict[str, Path]:
    """
    Split a COCO annotation file into train/val/test/unused subsets.

    Args:
        coco_file: Path to the source COCO annotation JSON file.
        output_dir: Directory to save the split annotation files.
        train_ratio: Fraction of data for training (default: 0.7).
        val_ratio: Fraction of data for validation (default: 0.15).
        test_ratio: Fraction of data for testing (default: 0.15).
        unused_ratio: Fraction of data to exclude (default: 0.0).
        seed: Random seed for reproducibility (default: 42).
        stratify_by_category: If True, maintain category distribution in splits.

    Returns:
        Dictionary mapping split names to output file paths.

    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    total_ratio = train_ratio + val_ratio + test_ratio + unused_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio}, unused={unused_ratio})"
        )

    coco_file = Path(coco_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load source COCO file
    with open(coco_file, "r") as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])
    categories = coco_data.get("categories", [])
    info = coco_data.get("info", {})
    licenses = coco_data.get("licenses", [])

    if seed is not None:
        random.seed(seed)

    # Get image IDs
    image_ids = [img["id"] for img in images]

    if stratify_by_category:
        # Group images by their primary category (most frequent)
        image_to_categories = defaultdict(list)
        for ann in annotations:
            image_to_categories[ann["image_id"]].append(ann["category_id"])

        # Assign each image to its most frequent category
        image_to_primary_cat = {}
        for img_id, cats in image_to_categories.items():
            if cats:
                image_to_primary_cat[img_id] = max(set(cats), key=cats.count)
            else:
                image_to_primary_cat[img_id] = None

        # Group images by category
        cat_to_images = defaultdict(list)
        for img_id in image_ids:
            cat = image_to_primary_cat.get(img_id)
            cat_to_images[cat].append(img_id)

        # Split each category group
        train_ids, val_ids, test_ids, unused_ids = [], [], [], []
        for cat, cat_image_ids in cat_to_images.items():
            random.shuffle(cat_image_ids)
            n = len(cat_image_ids)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            n_test = int(n * test_ratio)

            train_ids.extend(cat_image_ids[:n_train])
            val_ids.extend(cat_image_ids[n_train : n_train + n_val])
            test_ids.extend(cat_image_ids[n_train + n_val : n_train + n_val + n_test])
            unused_ids.extend(cat_image_ids[n_train + n_val + n_test :])
    else:
        # Simple random split
        random.shuffle(image_ids)
        n = len(image_ids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)

        train_ids = image_ids[:n_train]
        val_ids = image_ids[n_train : n_train + n_val]
        test_ids = image_ids[n_train + n_val : n_train + n_val + n_test]
        unused_ids = image_ids[n_train + n_val + n_test :]

    # Convert to sets for fast lookup
    splits = {
        "train": set(train_ids),
        "val": set(val_ids),
        "test": set(test_ids),
    }
    if unused_ratio > 0:
        splits["unused"] = set(unused_ids)

    # Build annotation lookup by image_id
    image_id_to_anns = defaultdict(list)
    for ann in annotations:
        image_id_to_anns[ann["image_id"]].append(ann)

    # Build image lookup by id
    image_id_to_img = {img["id"]: img for img in images}

    # Generate split files
    output_files = {}
    base_name = coco_file.stem

    for split_name, split_image_ids in splits.items():
        if not split_image_ids:
            continue

        split_images = [image_id_to_img[img_id] for img_id in split_image_ids]
        split_annotations = []
        for img_id in split_image_ids:
            split_annotations.extend(image_id_to_anns[img_id])

        split_data = {
            "info": info,
            "licenses": licenses,
            "categories": categories,
            "images": split_images,
            "annotations": split_annotations,
        }

        output_path = output_dir / f"{base_name}_{split_name}.json"
        with open(output_path, "w") as f:
            json.dump(split_data, f, indent=2)

        output_files[split_name] = output_path
        print(
            f"{split_name}: {len(split_images)} images, "
            f"{len(split_annotations)} annotations -> {output_path}"
        )

    return output_files


def merge_coco_datasets(
    coco_files: list[str | Path],
    output_file: str | Path,
    reindex: bool = True,
) -> Path:
    """
    Merge multiple COCO annotation files into one.

    Args:
        coco_files: List of paths to COCO annotation JSON files.
        output_file: Path for the merged output file.
        reindex: If True, reindex image and annotation IDs to avoid conflicts.

    Returns:
        Path to the merged output file.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    merged_images = []
    merged_annotations = []
    merged_categories = {}
    merged_info = {}
    merged_licenses = []

    image_id_offset = 0
    annotation_id_offset = 0

    for coco_file in coco_files:
        with open(coco_file, "r") as f:
            coco_data = json.load(f)

        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])

        if not merged_info:
            merged_info = coco_data.get("info", {})
        if not merged_licenses:
            merged_licenses = coco_data.get("licenses", [])

        # Add categories (deduplicate by id)
        for cat in categories:
            if cat["id"] not in merged_categories:
                merged_categories[cat["id"]] = cat

        if reindex:
            # Build old to new ID mapping
            old_to_new_image_id = {}
            for img in images:
                old_id = img["id"]
                new_id = image_id_offset + 1
                image_id_offset += 1
                old_to_new_image_id[old_id] = new_id
                img["id"] = new_id
                merged_images.append(img)

            for ann in annotations:
                annotation_id_offset += 1
                ann["id"] = annotation_id_offset
                ann["image_id"] = old_to_new_image_id[ann["image_id"]]
                merged_annotations.append(ann)
        else:
            merged_images.extend(images)
            merged_annotations.extend(annotations)

    merged_data = {
        "info": merged_info,
        "licenses": merged_licenses,
        "categories": list(merged_categories.values()),
        "images": merged_images,
        "annotations": merged_annotations,
    }

    with open(output_file, "w") as f:
        json.dump(merged_data, f, indent=2)

    print(
        f"Merged {len(coco_files)} files: "
        f"{len(merged_images)} images, {len(merged_annotations)} annotations "
        f"-> {output_file}"
    )

    return output_file


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Split COCO dataset into train/val/test/unused")
    subparsers = parser.add_subparsers(dest="command", help="Split method")

    # Ratio-based split
    ratio_parser = subparsers.add_parser("ratio", help="Split by ratios")
    ratio_parser.add_argument("coco_file", type=str, help="Path to source COCO annotation file")
    ratio_parser.add_argument("output_dir", type=str, help="Output directory for split files")
    ratio_parser.add_argument("--train", type=float, default=0.7, help="Train ratio (default: 0.7)")
    ratio_parser.add_argument("--val", type=float, default=0.15, help="Validation ratio (default: 0.15)")
    ratio_parser.add_argument("--test", type=float, default=0.15, help="Test ratio (default: 0.15)")
    ratio_parser.add_argument("--unused", type=float, default=0.0, help="Unused ratio (default: 0.0)")
    ratio_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    ratio_parser.add_argument("--stratify", action="store_true", help="Stratify by category")

    # CSV-based split
    csv_parser = subparsers.add_parser("csv", help="Split by CSV file")
    csv_parser.add_argument("coco_file", type=str, help="Path to source COCO annotation file")
    csv_parser.add_argument("csv_file", type=str, help="Path to CSV file with split assignments")
    csv_parser.add_argument("output_dir", type=str, help="Output directory for split files")
    csv_parser.add_argument("--file-col", type=str, default="file_name", help="Column name for file name (default: file_name)")
    csv_parser.add_argument("--split-col", type=str, default="split", help="Column name for split (default: split)")

    args = parser.parse_args()

    if args.command == "ratio":
        split_coco_dataset(
            coco_file=args.coco_file,
            output_dir=args.output_dir,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            unused_ratio=args.unused,
            seed=args.seed,
            stratify_by_category=args.stratify,
        )
    elif args.command == "csv":
        split_coco_from_csv(
            coco_file=args.coco_file,
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            file_name_column=args.file_col,
            split_column=args.split_col,
        )
    else:
        parser.print_help()
