from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import CNN
from utils import PROJECT_ROOT, OUTPUT_DIR, BATCH_SIZE, DEFAULT_NUM_WORKERS, DEVICE, IMAGE_EXTENSIONS, make_eval_transform


DEFAULT_MODEL_PATH = OUTPUT_DIR / "models" / "model2.pth"
DEFAULT_TEST_DIR = PROJECT_ROOT / "dataset" / "raw" / "test"
DEFAULT_TEST_LABELS_PATH = PROJECT_ROOT / "dataset" / "raw" / "test_labels.csv"
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "logs" / "test_results_v2.csv"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "logs" / "test_summary_v2.csv"


class TestDataset(Dataset):
    def __init__(
        self,
        test_dir: Path,
        test_labels_path: Path,
        transform: Callable[[Image.Image], torch.Tensor] | None = None,
    ) -> None:
        self.test_dir = test_dir
        self.transform = transform
        self.samples: list[tuple[Path, int | None]] = []

        label_map: dict[str, int] = {}
        if test_labels_path.exists():
            with test_labels_path.open("r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    file_id = row.get("id", "").strip()
                    label_raw = row.get("label", "").strip()
                    if not file_id or label_raw == "":
                        continue
                    try:
                        label_map[file_id] = int(float(label_raw))
                    except ValueError:
                        continue

        for img_path in sorted(test_dir.glob("*.*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label: int | None = None
            if img_path.name in label_map:
                label = label_map[img_path.name]
            else:
                for file_id, mapped_label in label_map.items():
                    if img_path.name in file_id:
                        label = mapped_label
                        break

            self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            if self.transform is not None:
                tensor = self.transform(image)
            else:
                tensor = make_eval_transform()(image)
        # -1 marks unlabeled samples
        return tensor, -1 if label is None else label, image_path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate v2 CNN model on test set.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained v2 model.")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Directory with test images.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_TEST_LABELS_PATH, help="CSV file with test labels.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for testing.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N batches.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="CSV output path for per-image results.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="CSV output path for summary metrics.")
    return parser.parse_args()


def make_loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())


def compute_scores(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "loss": 0.0,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "labeled_samples": float(total),
    }


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    log_interval: int,
) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()

    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    labeled_count = 0

    tp = tn = fp = fn = 0
    rows: list[dict[str, str]] = []

    total_batches = len(loader)

    with torch.no_grad():
        for batch_index, (images, labels, image_names) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            probabilities = torch.sigmoid(logits).view(-1)
            predictions = (probabilities >= 0.5).long()

            for image_name, label_value, prediction, probability in zip(
                image_names,
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
                probabilities.detach().cpu().tolist(),
            ):
                is_labeled = label_value in (0, 1)

                actual_outcome = "ai" if label_value == 1 else "real" if label_value == 0 else ""
                predicted_outcome = "ai" if prediction == 1 else "real"

                rows.append(
                    {
                        "image": image_name,
                        "actual_label": "" if not is_labeled else str(label_value),
                        "predicted_label": str(prediction),
                        "actual_outcome": actual_outcome,
                        "predicted_outcome": predicted_outcome,
                        "result": "" if not is_labeled else ("correct" if label_value == prediction else "incorrect"),
                        "confidence": f"{probability:.4f}",
                    }
                )

            labeled_mask = labels >= 0
            if labeled_mask.any():
                labeled_logits = logits[labeled_mask]
                labeled_labels = labels[labeled_mask].float().unsqueeze(1)
                loss = criterion(labeled_logits, labeled_labels)

                batch_labeled = int(labeled_mask.sum().item())
                labeled_count += batch_labeled
                running_loss += loss.item() * batch_labeled

                labeled_probs = torch.sigmoid(labeled_logits).view(-1)
                labeled_preds = (labeled_probs >= 0.5).long()
                labeled_targets = labeled_labels.long().view(-1)

                tp += int(((labeled_preds == 1) & (labeled_targets == 1)).sum().item())
                tn += int(((labeled_preds == 0) & (labeled_targets == 0)).sum().item())
                fp += int(((labeled_preds == 1) & (labeled_targets == 0)).sum().item())
                fn += int(((labeled_preds == 0) & (labeled_targets == 1)).sum().item())

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                avg_loss = running_loss / labeled_count if labeled_count else 0.0
                progress_pct = 100.0 * batch_index / max(1, total_batches)
                print(f"  Test batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}", flush=True)

    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / labeled_count if labeled_count else 0.0
    scores["labeled_samples"] = float(labeled_count)
    scores["total_samples"] = float(len(rows))
    return scores, rows


def save_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "actual_label", "predicted_label", "actual_outcome", "predicted_outcome", "result", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_summary(path: Path, scores: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric, value in scores.items():
            writer.writerow({"metric": metric, "value": f"{value:.6f}"})


def main() -> None:
    args = parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    transform = make_eval_transform()
    test_dataset = TestDataset(args.test_dir, args.labels_path, transform=transform)
    if len(test_dataset) == 0:
        raise ValueError(f"No valid test images found in {args.test_dir}")

    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)

    print(f"Test samples: {len(test_dataset)}")
    print(f"Device: {DEVICE}")
    print(f"DataLoader workers: {args.num_workers}")

    model = CNN().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    print("Running v2 test evaluation...")
    scores, rows = evaluate(model, test_loader, DEVICE, args.log_interval)

    print("\n" + "=" * 60)
    print("V2 Test Results:")
    print(f"Labeled Samples: {int(scores['labeled_samples'])}/{int(scores['total_samples'])}")
    print(f"Test Loss: {scores['loss']:.4f}")
    print(f"Test Accuracy: {scores['accuracy']:.4f}")
    print(f"Test Precision: {scores['precision']:.4f}")
    print(f"Test Recall: {scores['recall']:.4f}")
    print(f"Test F1: {scores['f1']:.4f}")
    print("=" * 60)

    save_results(args.results_path, rows)
    save_summary(args.summary_path, scores)

    print(f"Per-image results saved to {args.results_path}")
    print(f"Summary metrics saved to {args.summary_path}")


if __name__ == "__main__":
    main()
