"""Dataset classes and loading utilities."""

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from utils import IMAGE_EXTENSIONS, make_train_transform, make_eval_transform, pil_to_tensor, normalize_tensor, resize_image


@dataclass(frozen=True)
class SplitData:
    """Container for train/val split datasets."""

    train_dataset: Dataset
    val_dataset: Dataset
    class_names: list[str]


class ImageFolderDataset(Dataset):
    """Dataset that loads images from class folders."""

    def __init__(
        self, root: Path, transform: Callable[[Image.Image], torch.Tensor] | None = None, class_to_idx: dict[str, int] | None = None
    ) -> None:
        self.root = root
        self.transform = transform
        self.classes = sorted([entry.name for entry in root.iterdir() if entry.is_dir()]) if root.exists() else []
        if class_to_idx is not None:
            self.class_to_idx = class_to_idx
        else:
            if {"real", "ai"}.issubset(set(self.classes)):
                self.class_to_idx = {"real": 0, "ai": 1}
            else:
                self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
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
    """Split targets into train/val indices maintaining class distribution."""
    class_to_indices: dict[int, list[int]] = {}
    for index, class_index in enumerate(targets):
        class_to_indices.setdefault(int(class_index), []).append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for indices in class_to_indices.values():
        rng.shuffle(indices)
        val_count = max(1, int(round(len(indices) * val_fraction)))
        if val_count >= len(indices):
            val_count = max(1, len(indices) - 1)

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def compute_class_weights(targets: list[int]) -> torch.Tensor:
    """Compute class weights to handle imbalanced datasets."""
    unique_classes, class_counts = torch.unique(torch.tensor(targets), return_counts=True)
    total = len(targets)
    weights = total / (len(unique_classes) * class_counts.float())
    return weights


def load_datasets(data_root: Path, seed: int, val_fraction: float) -> tuple[SplitData, torch.Tensor]:
    """Load and split training dataset."""
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

    train_targets = [base_train_dataset.targets[i] for i in train_indices]
    class_weights = compute_class_weights(train_targets)

    return SplitData(train_dataset=train_dataset, val_dataset=val_dataset, class_names=class_names), class_weights
