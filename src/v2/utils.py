"""Utility functions and constants for image processing and model utilities."""

import os
from pathlib import Path
from typing import Callable

import torch
import random
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Constants
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2
SEED = 67
DEFAULT_NUM_WORKERS = max(0, min(8, (os.cpu_count() or 2) - 1))

NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dirs() -> None:
    """Create output directories if they don't exist."""
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "plots").mkdir(parents=True, exist_ok=True)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert PIL image to normalized tensor."""
    image = image.convert("RGB")
    raw_bytes = bytearray(image.tobytes())
    tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
    tensor = tensor.view(image.height, image.width, 3).permute(2, 0, 1).float() / 255.0
    return tensor


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor using ImageNet statistics."""
    return (tensor - NORMALIZE_MEAN) / NORMALIZE_STD


def resize_image(image: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """Resize image to specified size."""
    return image.resize((size, size), Image.Resampling.LANCZOS)


class TrainTransform:
    """Training augmentation: resize, optional flip, normalize."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize_image(image)
        if self.rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        tensor = pil_to_tensor(image)
        return normalize_tensor(tensor)


class EvalTransform:
    """Evaluation transform: resize and normalize without augmentation."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = resize_image(image)
        tensor = pil_to_tensor(image)
        return normalize_tensor(tensor)


def make_train_transform(seed: int) -> Callable[[Image.Image], torch.Tensor]:
    """Create training transform."""
    return TrainTransform(seed)


def make_eval_transform() -> Callable[[Image.Image], torch.Tensor]:
    """Create evaluation transform."""
    return EvalTransform()


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    """Create DataLoader with optimal settings."""
    if num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def compute_scores(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    """Compute classification metrics from confusion matrix."""
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
    """Convert logits to binary predictions and compute confusion matrix values."""
    predictions = (torch.sigmoid(logits) >= 0.5).long().view(-1)
    targets = labels.long().view(-1)

    tp = int(((predictions == 1) & (targets == 1)).sum().item())
    tn = int(((predictions == 0) & (targets == 0)).sum().item())
    fp = int(((predictions == 1) & (targets == 0)).sum().item())
    fn = int(((predictions == 0) & (targets == 1)).sum().item())

    return tp, tn, fp, fn
