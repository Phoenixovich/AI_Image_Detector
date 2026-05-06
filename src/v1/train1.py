from __future__ import annotations

import argparse
import csv
import random
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "models" / "model1.pth"
LOG_PATH = OUTPUT_DIR / "logs" / "training_log.csv"

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.2
SEED = 67
DEFAULT_NUM_WORKERS = max(0, min(8, (os.cpu_count() or 2) - 1))

NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass(frozen=True)
class SplitData:
    train_dataset: Dataset
    val_dataset: Dataset
    class_names: list[str]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN used for identifying AI-Generated Images.")
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT, help="Path to dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for Adam.")
    parser.add_argument("--val", type=float, default=VAL_FRACTION, help="Validation dataset fraction of training dataset.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducible splitting.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes for data loading.")
    parser.add_argument("--log-interval", type=int, default=200, help="Print training progress every N batches.")
    return parser.parse_args()

def ensure_output_dirs() -> None:
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)

# instead of using transforms.ToTensor()
def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB")
    bytes = bytearray(image.tobytes())
    tensor = torch.frombuffer(bytes, dtype=torch.uint8)
    tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return tensor

def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return (tensor - NORMALIZE_MEAN) / NORMALIZE_STD

def resize_image(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)

class TrainTransform:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize_image(image)
        if self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = pil_to_tensor(image)
        return normalize_tensor(tensor)


class EvalTransform:
    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize_image(image)
        tensor = pil_to_tensor(image)
        return normalize_tensor(tensor)


def make_train_transform(seed: int) -> Callable[[Image.Image], torch.Tensor]:
    return TrainTransform(seed)

def make_eval_transform() -> Callable[[Image.Image], torch.Tensor]:
    return EvalTransform()

class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, transform: Callable[[Image.Image], torch.Tensor] | None = None, class_to_idx: dict[str, int] | None = None) -> None:
        self.root = root
        self.transform = transform
        self.classes = sorted([entry.name for entry in root.iterdir() if entry.is_dir()]) if root.exists() else []
        self.class_to_idx = class_to_idx or {name: index for index, name in enumerate(self.classes)}
        self.samples: list[tuple[Path, int]] = []

        for class_name in self.classes:
            class_dir = root / class_name
            class_index = self.class_to_idx[class_name]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, class_index))

        self.targets = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            if self.transform is not None:
                tensor = self.transform(image)
            else: 
                tensor = normalize_tensor(pil_to_tensor(resize_image(image)))
        return tensor, label

def stratified_indices(targets: list[int], val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    class_to_indices: dict[int, list[int]] = {}
    for index, class_index in enumerate(targets):
        class_to_indices.setdefault(int(class_index), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in class_to_indices.values():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        if val_fraction >= len(indices):
            val_count = max(1, len(indices) - 1)
        
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])
    
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices

def load_datasets(data_root: Path, seed: int, val_fraction: float) -> SplitData:
    train_root = data_root / "train"

    if not train_root.exists():
        raise FileNotFoundError(f"Training folder not found: {train_root}")
    
    train_transform = make_train_transform(seed)
    eval_transform = make_eval_transform()

    base_train_dataset = ImageFolderDataset(train_root, transform=train_transform)
    if not base_train_dataset.classes:
        raise ValueError(f"No class folders found in {train_root}. Expected folders such as ai/ and real/.")
    
    class_names = base_train_dataset.classes

    train_indices, val_indices = stratified_indices(base_train_dataset.targets, val_fraction=val_fraction, seed=seed)

    eval_dataset = ImageFolderDataset(train_root, transform=eval_transform, class_to_idx=base_train_dataset.class_to_idx)
    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(eval_dataset, val_indices)

    return SplitData(train_dataset=train_dataset, val_dataset=val_dataset, class_names=class_names)


class CNN(nn.Module):
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
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)

def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    if num_workers > 0:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=True, prefetch_factor=2)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())

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
        "f1": f1
    }           

def accumulate_predictions(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int, int]:
    predictions = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    targets = labels.long().view(-1) 

    tp = int(((predictions == 1) & (targets == 1)).sum().item())
    tn = int(((predictions == 0) & (targets == 0)).sum().item())
    fp = int(((predictions == 1) & (targets == 0)).sum().item())
    fn = int(((predictions == 0) & (targets == 1)).sum().item())

    return tp, tn, fp, fn

def train_one_epoch(model, loader, criterion, optimizer, device, log_interval: int) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    tp = tn = fp = fn = 0
    sample_count = 0
    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size

        batch_tp, batch_tn, batch_fp, batch_fn = accumulate_predictions(logits, labels)
        tp += batch_tp
        tn += batch_tn
        fp += batch_fp
        fn += batch_fn

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
            avg_loss = running_loss / sample_count if sample_count else 0.0
            progress_pct = 100.0 * batch_index / max(1, total_batches)
            print(f"  Train batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}", flush=True)
        
    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / sample_count if sample_count else 0.0
    return scores

def evaluate(model, loader, criterion, device, log_interval: int) -> dict[str, float]:
    model.eval()

    running_loss = 0.0
    tp = tn = fp = fn = 0
    sample_count = 0
    total_batches = len(loader)

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

            batch_tp, batch_tn, batch_fp, batch_fn = accumulate_predictions(logits, labels)
            tp += batch_tp
            tn += batch_tn
            fp += batch_fp
            fn += batch_fn

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                avg_loss = running_loss / sample_count if sample_count else 0.0
                progress_pct = 100.0 * batch_index / max(1, total_batches)
                print(f"  Val   batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}", flush=True)

    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / sample_count if sample_count else 0.0
    return scores

def save_logs(rows: list[dict[str, float]]) -> None:
    with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"]
        )
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_output_dirs()

    print("Loading datasets...")

    split_data = load_datasets(args.data_root, args.seed, args.val)

    train_loader = make_loader(split_data.train_dataset, args.batch_size, shuffle=True, num_workers= args.num_workers)
    val_loader = make_loader(split_data.val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Datasets loaded.")

    print(f"Classes: {split_data.class_names}")
    print(f"Device: {DEVICE}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Train samples: {len(split_data.train_dataset)}")
    print(f"Val samples: {len(split_data.val_dataset)}")

    model = CNN().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_accuracy = 0.0
    best_model_weights: dict[str, torch.Tensor] | None = None
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}] started", flush=True)
        train_scores = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args.log_interval)
        val_scores = evaluate(model, val_loader, criterion, DEVICE, args.log_interval)

        print(f"Epoch [{epoch}/{args.epochs}] finished")
        print(f"Train loss: {train_scores['loss']:.4f} - Train acc: {train_scores['accuracy']:.4f} | Val loss: {val_scores['loss']:.4f} - Val acc: {val_scores['accuracy']:.4f}")
        print(f"Val precision: {val_scores['precision']:.4f} - Val recall: {val_scores['recall']:.4f} - Val f1: {val_scores['f1']:.4f}")

        print("-" * 10)

        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_scores["loss"],
                "train_accuracy": train_scores["accuracy"],
                "val_loss": val_scores["loss"],
                "val_accuracy": val_scores["accuracy"],
                "val_precision": val_scores["precision"],
                "val_recall": val_scores["recall"],
                "val_f1": val_scores["f1"],
            }
        )

        if val_scores["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_scores["accuracy"]
            best_model_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_model_weights, MODEL_PATH)
            print(f"Model saved to {MODEL_PATH}")

    save_logs(log_rows)
    print(f"Training finished. Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Logs saved to {LOG_PATH}")


if __name__ == "__main__":
    main()