from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFile, UnidentifiedImageError


# Allow Pillow to load some truncated images instead of failing hard.
ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize all images from raw/train and raw/test into processed/ while keeping the folder structure."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=SCRIPT_DIR,
        help="Folder that contains train/ and test/ (default: the raw/ folder containing this script).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRIPT_DIR.parent / "processed",
        help="Destination folder for processed data (default: sibling processed/ folder).",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("WIDTH", "HEIGHT"),
        help="Target image size (default: 256 256).",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def resize_image(source: Path, destination: Path, size: tuple[int, int]) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        with Image.open(source) as image:
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            resized = image.resize(size, Image.Resampling.LANCZOS)
            resized.save(destination)
        return True
    except (OSError, UnidentifiedImageError) as error:
        print(f"Skipping broken image: {source} ({error})")
        return False


def process_train_tree(input_root: Path, output_root: Path, size: tuple[int, int]) -> tuple[int, int]:
    processed = 0
    skipped = 0

    split_root = input_root / "train"
    if not split_root.exists():
        return processed, skipped

    for source in split_root.rglob("*"):
        if not source.is_file() or not is_image(source):
            continue

        relative_path = source.relative_to(input_root)
        destination = output_root / relative_path
        if resize_image(source, destination, size):
            processed += 1
        else:
            skipped += 1

    return processed, skipped


def process_test_flattened(input_root: Path, output_root: Path, size: tuple[int, int]) -> tuple[int, int]:
    processed = 0
    skipped = 0
    test_root = input_root / "test"
    target_root = output_root / "test"

    if not test_root.exists():
        return processed, skipped

    real_files = sorted(
        [path for path in (test_root / "real").rglob("*") if path.is_file() and is_image(path)],
        key=lambda p: p.name.lower(),
    ) if (test_root / "real").exists() else []

    fake_files = sorted(
        [path for path in (test_root / "fake").rglob("*") if path.is_file() and is_image(path)],
        key=lambda p: p.name.lower(),
    ) if (test_root / "fake").exists() else []

    for index, source in enumerate(real_files, start=1):
        destination = target_root / f"aa{index:05d}{source.suffix.lower()}"
        if resize_image(source, destination, size):
            processed += 1
        else:
            skipped += 1

    for index, source in enumerate(fake_files, start=1):
        destination = target_root / f"ff{index:05d}{source.suffix.lower()}"
        if resize_image(source, destination, size):
            processed += 1
        else:
            skipped += 1

    return processed, skipped


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    size = tuple(args.size)

    train_processed, train_skipped = process_train_tree(input_root, output_root, size)
    test_processed, test_skipped = process_test_flattened(input_root, output_root, size)
    processed = train_processed + test_processed
    skipped = train_skipped + test_skipped

    print(f"Resized {processed} image(s) into {output_root}.")
    print(f"  train/: {train_processed}")
    print(f"  test/ (flattened aa*/ff*): {test_processed}")
    if skipped:
        print(f"Skipped {skipped} broken/unreadable image(s).")


if __name__ == "__main__":
    main()
