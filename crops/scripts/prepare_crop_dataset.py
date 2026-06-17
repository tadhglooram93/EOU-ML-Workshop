"""
Prepare a two-class crop image dataset for the farm image-classification workshop.

Current classes:
    - alfalfa
    - garbanzo

Run from the repo root:

    python crops/scripts/prepare_crop_dataset.py

Expected input folder (paths relative to repo root):

    crops/data/raw_crop_images/
        alfalfa/
            img/
                image1.jpg
                image2.jpg
                ...
        chickpeas/
            img/
                image1.jpg
                image2.jpg
                ...

The script will center-crop each image to a square, resize it, and create:

    crops/data/crop_images/
        train/
            alfalfa/
            garbanzo/
        val/
            alfalfa/
            garbanzo/
        test/
            alfalfa/
            garbanzo/
        dataset_manifest.csv

This folder structure works with:
    - Keras image_dataset_from_directory
    - Google Teachable Machine class-folder uploads
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image, ImageOps, UnidentifiedImageError


DEFAULT_CLASS_ALIASES: Dict[str, List[str]] = {
    "alfalfa": ["alfalfa", "alfafa", "lucerne"],
    "garbanzo": ["garbanzo", "garbanzo_beans", "garbanzo beans", "chickpea", "chickpeas"],
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = "crops/data/raw_crop_images"
DEFAULT_OUTPUT_DIR = "crops/data/crop_images"


def repo_path(relative: str | Path) -> Path:
    return REPO_ROOT / relative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare crop images for a two-class classifier.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Raw crop image folder relative to repo root",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Prepared dataset folder relative to repo root",
    )
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Delete output folder before writing.")
    parser.add_argument("--max-per-class", type=int, default=None, help="Optional cap for quick tests.")
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return name.lower().strip().replace("_", " ").replace("-", " ")


def find_class_dirs(input_dir: Path) -> Dict[str, Path]:
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input folder not found: {input_dir}\n"
            f"Create folders like {DEFAULT_INPUT_DIR}/alfalfa and {DEFAULT_INPUT_DIR}/chickpeas."
        )

    child_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    normalized_to_path = {normalize_name(p.name): p for p in child_dirs}

    found: Dict[str, Path] = {}

    for clean_label, aliases in DEFAULT_CLASS_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_name(alias)
            if normalized_alias in normalized_to_path:
                found[clean_label] = normalized_to_path[normalized_alias]
                break

        if clean_label not in found:
            available = ", ".join(sorted(p.name for p in child_dirs)) or "none"
            raise FileNotFoundError(
                f"Could not find a folder for class '{clean_label}'.\n"
                f"Looked for aliases: {aliases}\n"
                f"Available folders in {input_dir}: {available}"
            )

    return found


def list_images(folder: Path) -> List[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def center_crop_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    right = left + side
    bottom = top + side
    return image.crop((left, top, right, bottom))


def process_image(src: Path, dst: Path, image_size: int, jpeg_quality: int) -> Tuple[int, int] | None:
    try:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            original_size = img.size
            img = img.convert("RGB")
            img = center_crop_square(img)
            img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst, format="JPEG", quality=jpeg_quality, optimize=True)
            return original_size
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        print(f"Skipping unreadable image: {src} ({exc})")
        return None


def split_paths(paths: List[Path], train_frac: float, val_frac: float, seed: int, max_per_class: int | None) -> Dict[str, List[Path]]:
    paths = paths.copy()
    rng = random.Random(seed)
    rng.shuffle(paths)

    if max_per_class is not None:
        paths = paths[:max_per_class]

    n = len(paths)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }


def validate_split_fractions(train_frac: float, val_frac: float, test_frac: float) -> None:
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Train/val/test fractions must sum to 1. Got {total}")


def clear_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def write_manifest(manifest_path: Path, rows: Iterable[dict]) -> None:
    fieldnames = [
        "split",
        "class_name",
        "source_path",
        "output_path",
        "original_width",
        "original_height",
        "processed_width",
        "processed_height",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    validate_split_fractions(args.train_frac, args.val_frac, args.test_frac)

    input_dir = repo_path(args.input_dir)
    output_dir = repo_path(args.output_dir)
    clear_output_dir(output_dir, args.overwrite)

    class_dirs = find_class_dirs(input_dir)
    manifest_rows: List[dict] = []
    summary: Dict[str, Dict[str, int]] = {}

    for class_name, folder in class_dirs.items():
        images = list_images(folder)
        if not images:
            raise ValueError(f"No images found in {folder}")

        split_map = split_paths(
            paths=images,
            train_frac=args.train_frac,
            val_frac=args.val_frac,
            seed=args.seed,
            max_per_class=args.max_per_class,
        )

        summary[class_name] = {}

        for split_name, split_images in split_map.items():
            summary[class_name][split_name] = 0
            for i, src in enumerate(split_images):
                dst = output_dir / split_name / class_name / f"{class_name}_{i:04d}.jpg"
                original_size = process_image(src, dst, args.image_size, args.jpeg_quality)
                if original_size is None:
                    continue

                summary[class_name][split_name] += 1
                manifest_rows.append({
                    "split": split_name,
                    "class_name": class_name,
                    "source_path": str(src),
                    "output_path": str(dst),
                    "original_width": original_size[0],
                    "original_height": original_size[1],
                    "processed_width": args.image_size,
                    "processed_height": args.image_size,
                })

    write_manifest(output_dir / "dataset_manifest.csv", manifest_rows)

    print("\nDone. Prepared dataset:")
    print(output_dir)
    print("\nCounts:")
    for class_name, counts in summary.items():
        print(f"  {class_name}")
        for split_name in ["train", "val", "test"]:
            print(f"    {split_name}: {counts.get(split_name, 0)}")

    print("\nNext steps:")
    print(f"  1. Use {DEFAULT_OUTPUT_DIR}/train and {DEFAULT_OUTPUT_DIR}/val in Keras.")
    print("  2. Upload class folders to Teachable Machine if you want the no-code demo.")


if __name__ == "__main__":
    main()
