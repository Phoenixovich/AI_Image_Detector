from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "raw"
TEST_DIR = DATASET_ROOT / "test"
TEST_LABELS_PATH = DATASET_ROOT / "test_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "models" / "model1.pth"
RESULTS_PATH = OUTPUT_DIR / "logs" / "test_results1.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 32
DEFAULT_NUM_WORKERS = 4

NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN model on test set.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH, help="Path to trained model.")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to test images.")
    parser.add_argument("--labels-path", type=Path, default=TEST_LABELS_PATH, help="Path to test labels CSV.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--log-interval", type=int, default=200, help="Print progress every N batches.")
    return parser.parse_args()


def ensure_output_dirs() -> None:
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    raw_bytes = bytearray(image.tobytes())
    tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
    tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return tensor


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - NORMALIZE_MEAN) / NORMALIZE_STD


def resize_image(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


class EvalTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize_image(image)
        tensor = pil_to_tensor(image)
        return normalize_tensor(tensor)


def make_eval_transform() -> Callable[[Image.Image], torch.Tensor]:
    return EvalTransform()


class TestDataset(Dataset):
    def __init__(self, test_dir: Path, labels_csv: Path, transform: Callable[[Image.Image], torch.Tensor] | None = None) -> None:
        self.test_dir = test_dir
        self.transform = transform
        self.samples: list[tuple[Path, int | None]] = []

        # Load labels from CSV
        label_map: dict[str, int] = {}
        try:
            with labels_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_id = row["id"]
                    label = float(row["label"])
                    label_map[file_id] = int(label)
        except Exception as e:
            print(f"Warning: Could not load labels from {labels_csv}: {e}")

        # Find all images in test directory
        image_paths = sorted(test_dir.glob("*.*"))
        for img_path in image_paths:
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
                # Try to find label by filename
                label = None
                for file_id in label_map:
                    if img_path.name in file_id:
                        label = label_map[file_id]
                        break

                self.samples.append((img_path, label))

        if not self.samples:
            print(f"Warning: No images found in {test_dir}")
        else:
            labeled_count = sum(1 for _, label in self.samples if label is not None)
            unlabeled_count = len(self.samples) - labeled_count
            print(f"Loaded {len(self.samples)} test images ({labeled_count} labeled, {unlabeled_count} unlabeled).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            if self.transform is not None:
                tensor = self.transform(image)
            else:
                tensor = normalize_tensor(pil_to_tensor(resize_image(image)))
        return tensor, -1 if label is None else label, path.name


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


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
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def accumulate_predictions(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int, int]:
    predictions = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    targets = labels.long().view(-1)

    true_positive = int(((predictions == 1) & (targets == 1)).sum().item())
    true_negative = int(((predictions == 0) & (targets == 0)).sum().item())
    false_positive = int(((predictions == 1) & (targets == 0)).sum().item())
    false_negative = int(((predictions == 0) & (targets == 1)).sum().item())

    return true_positive, true_negative, false_positive, false_negative


def evaluate(model, loader, device, log_interval: int) -> tuple[dict[str, float], list[dict[str, str]]]:
    model.eval()

    running_loss = 0.0
    tp = tn = fp = fn = 0
    sample_count = 0
    total_batches = len(loader)
    criterion = nn.BCEWithLogitsLoss()
    per_image_rows: list[dict[str, str]] = []

    with torch.no_grad():
        for batch_index, (images, labels, image_names) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            labeled_mask = labels >= 0
            if labeled_mask.any():
                labeled_logits = logits[labeled_mask]
                labeled_targets = labels[labeled_mask].float().unsqueeze(1)
                loss = criterion(labeled_logits, labeled_targets)

                batch_size = int(labeled_mask.sum().item())
                running_loss += loss.item() * batch_size
                sample_count += batch_size

                batch_tp, batch_tn, batch_fp, batch_fn = accumulate_predictions(labeled_logits, labeled_targets)
                tp += batch_tp
                tn += batch_tn
                fp += batch_fp
                fn += batch_fn
            else:
                loss = None

            probabilities = torch.sigmoid(logits).view(-1).detach().cpu()
            predicted_labels = (probabilities >= 0.5).long().tolist()
            actual_labels = labels.detach().cpu().tolist()

            for image_name, actual_label, predicted_label, probability in zip(image_names, actual_labels, predicted_labels, probabilities.tolist()):
                actual_outcome = "ai" if actual_label == 1 else "real" if actual_label == 0 else ""
                predicted_outcome = "ai" if predicted_label == 1 else "real"
                label_text = "" if actual_label == -1 else str(actual_label)
                per_image_rows.append(
                    {
                        "image": image_name,
                        "actual_label": label_text,
                        "predicted_label": str(predicted_label),
                        "actual_outcome": actual_outcome,
                        "predicted_outcome": predicted_outcome,
                        "result": "correct" if actual_label in (0, 1) and actual_label == predicted_label else ("incorrect" if actual_label in (0, 1) else ""),
                        "confidence": f"{probability:.4f}",
                    }
                )

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                avg_loss = running_loss / sample_count if sample_count else 0.0
                progress_pct = 100.0 * batch_index / max(1, total_batches)
                print(f"  Test batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}", flush=True)

    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / sample_count if sample_count else 0.0
    return scores, per_image_rows


def save_results(rows: list[dict[str, str]]) -> None:
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "actual_label", "predicted_label", "actual_outcome", "predicted_outcome", "result", "confidence"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    ensure_output_dirs()

    # Validate paths
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")

    print("Loading test dataset...")
    eval_transform = make_eval_transform()
    test_dataset = TestDataset(args.test_dir, args.labels_path, transform=eval_transform)

    if len(test_dataset) == 0:
        print("Error: No test samples loaded. Exiting.")
        return

    test_loader = make_loader(test_dataset, args.batch_size, args.num_workers)

    print(f"Test dataset loaded: {len(test_dataset)} samples")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {args.batch_size}")
    print(f"DataLoader workers: {args.num_workers}")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model = SimpleCNN().to(DEVICE)
    state_dict = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Model loaded.")

    # Evaluate
    print("Running evaluation...")
    scores, rows = evaluate(model, test_loader, DEVICE, args.log_interval)

    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Loss:      {scores['loss']:.4f}")
    print(f"Accuracy:  {scores['accuracy']:.4f}")
    print(f"Precision: {scores['precision']:.4f}")
    print(f"Recall:    {scores['recall']:.4f}")
    print(f"F1-Score:  {scores['f1']:.4f}")
    print("=" * 60)

    save_results(rows)
    print(f"Per-image results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
