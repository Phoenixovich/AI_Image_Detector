from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "logs" / "test_results_v6.csv"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "dataset" / "processed" / "train"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "processed" / "isolate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy false positives and false negatives into isolate folders.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="CSV file with per-image results.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR, help="Directory that contains the source images.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where fp/fn folders will be created.")
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


def isolate_errors(results_path: Path, source_dir: Path, output_dir: Path) -> tuple[int, int, int]:
    rows = read_results(results_path)
    fp_dir = output_dir / "fp"
    fn_dir = output_dir / "fn"
    output_dir.mkdir(parents=True, exist_ok=True)

    fp_count = 0
    fn_count = 0
    skipped_count = 0

    for row in rows:
        image_name = (row.get("image") or "").strip()
        if not image_name:
            skipped_count += 1
            continue

        if is_false_positive(row):
            source_path = find_source_image(source_dir, image_name)
            if source_path is None:
                skipped_count += 1
                continue
            safe_copy(source_path, fp_dir)
            fp_count += 1
        elif is_false_negative(row):
            source_path = find_source_image(source_dir, image_name)
            if source_path is None:
                skipped_count += 1
                continue
            safe_copy(source_path, fn_dir)
            fn_count += 1

    return fp_count, fn_count, skipped_count


def main() -> None:
    args = parse_args()

    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    fp_count, fn_count, skipped_count = isolate_errors(args.results_path, args.source_dir, args.output_dir)

    print(f"False positives copied: {fp_count}")
    print(f"False negatives copied: {fn_count}")
    print(f"Skipped rows/files: {skipped_count}")
    print(f"Output folder: {args.output_dir}")


if __name__ == "__main__":
    main()
