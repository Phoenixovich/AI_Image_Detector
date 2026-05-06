from __future__ import annotations

import argparse
import csv
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
	script_dir = Path(__file__).resolve().parent

	parser = argparse.ArgumentParser(
		description="Create a CSV for test images with labels: real=0, fake=1."
	)
	parser.add_argument(
		"--test-dir",
		type=Path,
		default=script_dir / "test",
		help="Path to the test folder containing real/ and fake/.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=script_dir / "test_labels.csv",
		help="Path to output CSV file.",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Recursively scan files inside real/ and fake/.",
	)
	return parser.parse_args()


def list_image_files(folder: Path, recursive: bool) -> list[Path]:
	iterator = folder.rglob("*") if recursive else folder.glob("*")
	files = [path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
	return sorted(files, key=lambda p: p.name.lower())


def build_rows(test_dir: Path, recursive: bool) -> list[dict[str, str]]:
	real_dir = test_dir / "real"
	fake_dir = test_dir / "fake"

	if not real_dir.exists():
		raise FileNotFoundError(f"Missing folder: {real_dir}")
	if not fake_dir.exists():
		raise FileNotFoundError(f"Missing folder: {fake_dir}")

	rows: list[dict[str, str]] = []

	for image_path in list_image_files(real_dir, recursive):
		rows.append({"id": "aa"+image_path.name, "label": "0"})

	for image_path in list_image_files(fake_dir, recursive):
		rows.append({"id": "ff"+image_path.name, "label": "1"})

	return rows


def write_csv(output_path: Path, rows: list[dict[str, str]]) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with output_path.open("w", newline="", encoding="utf-8") as file:
		writer = csv.DictWriter(file, fieldnames=["id", "label"])
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	args = parse_args()
	rows = build_rows(args.test_dir, args.recursive)
	write_csv(args.output, rows)

	real_count = sum(1 for row in rows if row["label"] == "0")
	fake_count = sum(1 for row in rows if row["label"] == "1")

	print(f"CSV written to: {args.output}")
	print(f"Real (0): {real_count}")
	print(f"Fake (1): {fake_count}")
	print(f"Total: {len(rows)}")


if __name__ == "__main__":
	main()
