from __future__ import annotations

import argparse
import csv
import colorsys
import math
import shutil
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "logs" / "test_results_final.csv"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "dataset" / "processed" / "test"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "featuresort"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sort images into feature buckets for visual analysis.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="CSV file with per-image results.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Directory that contains the source images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where feature buckets will be created.")
    parser.add_argument(
        "--mode",
        choices=("errors", "all"),
        default="errors",
        help="Sort only misclassified images or every image in the CSV.",
    )
    return parser.parse_args()


def read_results(results_path: Path) -> list[dict[str, str]]:
    if not results_path.exists():
        raise FileNotFoundError(f"Results CSV not found: {results_path}")

    with results_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


def is_false_positive(row: dict[str, str]) -> bool:
    actual = (row.get("actual_label") or "").strip()
    predicted = (row.get("predicted_label") or "").strip()
    return actual == "0" and predicted == "1"


def is_false_negative(row: dict[str, str]) -> bool:
    actual = (row.get("actual_label") or "").strip()
    predicted = (row.get("predicted_label") or "").strip()
    return actual == "1" and predicted == "0"


def get_error_type(row: dict[str, str]) -> str:
    if is_false_positive(row):
        return "fp"
    if is_false_negative(row):
        return "fn"
    return "correct"


def should_include_row(row: dict[str, str], mode: str) -> bool:
    if mode == "all":
        return True
    return get_error_type(row) in {"fp", "fn"}


def find_source_image(source_dir: Path, image_name: str) -> Path | None:
    exact_match = source_dir / image_name
    if exact_match.exists():
        return exact_match

    matches = [path for path in source_dir.rglob(image_name) if path.is_file()]
    if matches:
        return matches[0]

    return None


def safe_copy(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination_path = destination_dir / source_path.name

    if not destination_path.exists():
        shutil.copy2(source_path, destination_path)
        return destination_path

    stem = source_path.stem
    suffix = source_path.suffix
    counter = 1
    while True:
        candidate = destination_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            shutil.copy2(source_path, candidate)
            return candidate
        counter += 1


def compute_image_features(image_path: Path) -> tuple[float, float, float]:
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((64, 64))
        pixels = list(image.getdata())

    if not pixels:
        return 0.0, 0.0, 0.0

    luminance_values: list[float] = []
    saturation_values: list[float] = []

    for red, green, blue in pixels:
        red_value = red / 255.0
        green_value = green / 255.0
        blue_value = blue / 255.0
        _, saturation, _ = colorsys.rgb_to_hsv(red_value, green_value, blue_value)
        luminance = 0.299 * red_value + 0.587 * green_value + 0.114 * blue_value
        luminance_values.append(luminance)
        saturation_values.append(saturation)

    brightness = sum(luminance_values) / len(luminance_values)
    contrast = math.sqrt(sum((value - brightness) ** 2 for value in luminance_values) / len(luminance_values))
    saturation = sum(saturation_values) / len(saturation_values)
    return brightness, contrast, saturation


def brightness_bucket(brightness: float) -> str:
    if brightness < 0.35:
        return "dark"
    if brightness < 0.65:
        return "mid"
    return "bright"


def contrast_bucket(contrast: float) -> str:
    if contrast < 0.16:
        return "smooth"
    if contrast < 0.28:
        return "mixed"
    return "high_contrast"


def saturation_bucket(saturation: float) -> str:
    if saturation < 0.25:
        return "muted"
    if saturation < 0.55:
        return "balanced"
    return "vivid"


def bucket_name(brightness: float, contrast: float, saturation: float) -> str:
    return f"{brightness_bucket(brightness)}__{contrast_bucket(contrast)}__{saturation_bucket(saturation)}"


def sort_by_features(results_path: Path, source_dir: Path, output_dir: Path, mode: str) -> tuple[int, int, int]:
    rows = read_results(results_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "feature_sort_summary.csv"

    written_rows: list[dict[str, str]] = []
    copied_count = 0
    skipped_count = 0

    for row in rows:
        image_name = (row.get("image") or "").strip()
        if not image_name or not should_include_row(row, mode):
            continue

        source_path = find_source_image(source_dir, image_name)
        if source_path is None:
            skipped_count += 1
            continue

        try:
            brightness, contrast, saturation = compute_image_features(source_path)
        except OSError:
            skipped_count += 1
            continue

        error_type = get_error_type(row)
        feature_bucket = bucket_name(brightness, contrast, saturation)
        destination_dir = output_dir / error_type / feature_bucket
        copied_path = safe_copy(source_path, destination_dir)
        copied_count += 1

        written_rows.append(
            {
                "image": image_name,
                "error_type": error_type,
                "brightness": f"{brightness:.4f}",
                "contrast": f"{contrast:.4f}",
                "saturation": f"{saturation:.4f}",
                "feature_bucket": feature_bucket,
                "destination": str(copied_path.relative_to(output_dir)),
                "actual_label": (row.get("actual_label") or "").strip(),
                "predicted_label": (row.get("predicted_label") or "").strip(),
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "image",
            "error_type",
            "brightness",
            "contrast",
            "saturation",
            "feature_bucket",
            "destination",
            "actual_label",
            "predicted_label",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written_rows)

    return copied_count, skipped_count, len(written_rows)


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    copied_count, skipped_count, summarized_count = sort_by_features(
        args.results_path,
        args.source_dir,
        args.output_dir,
        args.mode,
    )

    print(f"Images copied: {copied_count}")
    print(f"Rows summarized: {summarized_count}")
    print(f"Skipped rows/files: {skipped_count}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
