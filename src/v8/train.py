from __future__ import annotations

import argparse
import csv
import json
import math
import random
from time import perf_counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler

from model import CNN
from dataset import load_datasets
from utils import (
    DEVICE,
    DATASET_ROOT,
    OUTPUT_DIR,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    VAL_FRACTION,
    IMAGE_EXTENSIONS,
    SEED,
    DEFAULT_NUM_WORKERS,
    make_loader,
    make_eval_transform,
    ensure_output_dirs,
    compute_scores,
    confusion_from_predictions,
    format_elapsed,
)

MODEL_PATH = OUTPUT_DIR / "models" / "model8.pth"
THRESHOLD_PATH = OUTPUT_DIR / "models" / "model8_threshold.json"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
LOG_PATH = OUTPUT_DIR / "logs" / f"training_log_v8_{RUN_TIMESTAMP}.csv"
PLOT_PATH = OUTPUT_DIR / "plots" / f"training_overfit_v8_{RUN_TIMESTAMP}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v8 gated multi-evidence CNN with auxiliary branch losses.")
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT, help="Path to dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 regularization weight decay.")
    parser.add_argument("--val", type=float, default=VAL_FRACTION, help="Validation dataset fraction.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--log-interval", type=int, default=200, help="Print progress every N batches.")
    parser.add_argument("--max-pos-weight", type=float, default=1.5, help="Upper cap for BCE positive class weight.")
    parser.add_argument("--min-pos-weight", type=float, default=0.67, help="Lower cap for BCE positive class weight.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="Stop if val selection score does not improve for this many epochs.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.001,
        help="Minimum val selection score improvement required to reset patience.",
    )
    parser.add_argument(
        "--hard-examples-dir",
        type=Path,
        default=None,
        help="Optional folder containing hard examples (e.g. dataset/processed/isolate) to upweight by filename.",
    )
    parser.add_argument(
        "--hard-examples-csv",
        type=Path,
        default=None,
        help="Optional CSV with an 'image' column listing hard example file names to upweight.",
    )
    parser.add_argument(
        "--hard-weight-multiplier",
        type=float,
        default=3.0,
        help="Sampling weight multiplier for hard examples (>=1.0).",
    )
    parser.add_argument(
        "--dynamic-hard-mining",
        action="store_true",
        help="Recompute hard examples from train-split misclassifications after each epoch.",
    )
    parser.add_argument(
        "--dynamic-hard-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to mine train-split hard examples.",
    )
    parser.add_argument(
        "--hard-mining-workers",
        type=int,
        default=0,
        help="DataLoader workers for dynamic hard mining (0 is safest on Windows).",
    )
    parser.add_argument("--focal-gamma", type=float, default=1.0, help="Focal loss gamma parameter.")
    parser.add_argument("--focal-alpha", type=float, default=0.5, help="Focal loss alpha parameter for positive class balancing.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs before cosine decay.")
    parser.add_argument("--min-lr-ratio", type=float, default=0.1, help="Minimum LR ratio during cosine decay.")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="Exponential moving average decay for model weights.")
    parser.add_argument("--rgb-loss-weight", type=float, default=0.20, help="Auxiliary RGB branch loss weight.")
    parser.add_argument("--residual-loss-weight", type=float, default=0.10, help="Auxiliary residual branch loss weight.")
    return parser.parse_args()


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else torch.tensor(1.0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=self.pos_weight)
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * torch.pow(1.0 - pt, self.gamma)
        return (focal_weight * bce).mean()


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name] = parameter.detach().clone()

    def update(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        self.backup = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            self.backup[name] = parameter.detach().clone()
            parameter.data.copy_(self.shadow[name].data)

    def restore(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name in self.backup:
                parameter.data.copy_(self.backup[name].data)
        self.backup = {}


def unpack_model_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["logits"]
    return output


def compute_v8_loss(
    output: torch.Tensor | dict[str, torch.Tensor],
    labels: torch.Tensor,
    criterion: nn.Module,
    rgb_loss_weight: float,
    residual_loss_weight: float,
) -> torch.Tensor:
    if not isinstance(output, dict):
        return criterion(output, labels)

    loss = criterion(output["logits"], labels)
    if rgb_loss_weight > 0.0:
        loss = loss + rgb_loss_weight * criterion(output["rgb_logits"], labels)
    if residual_loss_weight > 0.0:
        loss = loss + residual_loss_weight * criterion(output["residual_logits"], labels)
    return loss


def checkpoint_score(scores: dict[str, float]) -> float:
    return 0.5 * scores["f1"] + 0.5 * scores["balanced_accuracy"]


def make_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_ratio: float,
) -> optim.lr_scheduler.LambdaLR:
    total = max(1, int(total_epochs))
    warmup = max(0, min(int(warmup_epochs), total - 1))
    min_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

    def lr_lambda(epoch: int) -> float:
        if warmup > 0 and epoch < warmup:
            return float(epoch + 1) / float(warmup)
        progress = (epoch - warmup) / max(1, total - warmup - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_training_plots(rows: list[dict[str, float]], plot_path: Path, start_time: float) -> None:
    if not rows:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[{format_elapsed(perf_counter() - start_time)}] Skipping training plots (matplotlib unavailable): {exc}", flush=True)
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) for row in rows]
    train_acc = [float(row["train_accuracy"]) for row in rows]
    val_acc = [float(row["val_accuracy"]) for row in rows]

    val_precision = [float(row["val_precision"]) for row in rows]
    val_recall = [float(row["val_recall"]) for row in rows]
    val_specificity = [float(row["val_specificity"]) for row in rows]
    val_f1 = [float(row["val_f1"]) for row in rows]
    val_balanced = [float(row["val_balanced_accuracy"]) for row in rows]
    val_threshold = [float(row["val_threshold"]) for row in rows]

    loss_gap = [v - t for t, v in zip(train_loss, val_loss)]
    acc_gap = [t - v for t, v in zip(train_acc, val_acc)]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, label="train_loss")
    ax.plot(epochs, val_loss, label="val_loss")
    ax.set_title("Loss vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, train_acc, label="train_accuracy")
    ax.plot(epochs, val_acc, label="val_accuracy")
    ax.set_title("Accuracy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 2]
    ax.plot(epochs, loss_gap, label="val_loss - train_loss")
    ax.plot(epochs, acc_gap, label="train_acc - val_acc")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_title("Generalization Gaps")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, val_precision, label="val_precision")
    ax.plot(epochs, val_recall, label="val_recall")
    ax.plot(epochs, val_specificity, label="val_specificity")
    ax.plot(epochs, val_f1, label="val_f1")
    ax.set_title("Validation Metrics")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, val_balanced, label="val_balanced_accuracy")
    ax.set_title("Validation Balanced Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 2]
    ax.plot(epochs, val_threshold, label="val_threshold")
    ax.set_title("Chosen Threshold per Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("V8 Training Diagnostics (Overfitting / Underfitting)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


def log_message(start_time: float, message: str) -> None:
    elapsed = format_elapsed(perf_counter() - start_time)
    print(f"[{elapsed}] {message}", flush=True)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
    start_time: float,
    ema: ExponentialMovingAverage | None = None,
    rgb_loss_weight: float = 0.20,
    residual_loss_weight: float = 0.10,
) -> dict[str, float]:
    model.train()

    running_loss = 0.0
    tp = tn = fp = fn = 0
    sample_count = 0
    total_batches = len(loader)

    for batch_index, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        output = model(images, return_aux=True)
        logits = unpack_model_output(output)
        loss = compute_v8_loss(output, labels, criterion, rgb_loss_weight, residual_loss_weight)
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)

        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= 0.5).long()
        targets = labels.long().view(-1)
        batch_tp, batch_tn, batch_fp, batch_fn = confusion_from_predictions(preds, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
        tp += batch_tp
        tn += batch_tn
        fp += batch_fp
        fn += batch_fn

        if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
            avg_loss = running_loss / sample_count if sample_count else 0.0
            progress_pct = 100.0 * batch_index / max(1, total_batches)
            log_message(start_time, f"  Train batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}")

    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / sample_count if sample_count else 0.0
    scores["tp"] = float(tp)
    scores["tn"] = float(tn)
    scores["fp"] = float(fp)
    scores["fn"] = float(fn)
    return scores


def evaluate_with_threshold_search(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    log_interval: int,
    start_time: float,
    rgb_loss_weight: float = 0.20,
    residual_loss_weight: float = 0.10,
) -> tuple[dict[str, float], float]:
    model.eval()

    running_loss = 0.0
    sample_count = 0
    total_batches = len(loader)

    all_probs: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            output = model(images, return_aux=True)
            logits = unpack_model_output(output)
            loss = compute_v8_loss(output, labels, criterion, rgb_loss_weight, residual_loss_weight)

            probs = torch.sigmoid(logits).view(-1).detach().cpu()
            targets = labels.long().view(-1).detach().cpu()
            all_probs.append(probs)
            all_targets.append(targets)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            sample_count += batch_size

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                avg_loss = running_loss / sample_count if sample_count else 0.0
                progress_pct = 100.0 * batch_index / max(1, total_batches)
                log_message(start_time, f"  Val   batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}")

    probs = torch.cat(all_probs) if all_probs else torch.empty(0)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

    best_threshold = 0.5
    best_selection_score = -1.0
    best_conf = (0, 0, 0, 0)

    for step in range(5, 100):
        threshold = step / 100.0
        preds = (probs >= threshold).long()
        tp, tn, fp, fn = confusion_from_predictions(preds, targets)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_acc = 0.5 * (tpr + tnr)
        scores = compute_scores(tp, tn, fp, fn)
        scores["balanced_accuracy"] = balanced_acc
        selection_score = checkpoint_score(scores)

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_threshold = threshold
            best_conf = (tp, tn, fp, fn)

    tp, tn, fp, fn = best_conf
    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / sample_count if sample_count else 0.0
    scores["balanced_accuracy"] = 0.5 * (scores["recall"] + scores["specificity"])
    scores["selection_score"] = checkpoint_score(scores)
    scores["tp"] = float(tp)
    scores["tn"] = float(tn)
    scores["fp"] = float(fp)
    scores["fn"] = float(fn)

    return scores, best_threshold


def collect_hard_example_names(hard_examples_dir: Path | None, hard_examples_csv: Path | None) -> set[str]:
    hard_names: set[str] = set()

    if hard_examples_dir is not None and hard_examples_dir.exists():
        for path in hard_examples_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                hard_names.add(path.name)

    if hard_examples_csv is not None and hard_examples_csv.exists():
        with hard_examples_csv.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                image_name = (row.get("image") or "").strip()
                if image_name:
                    hard_names.add(Path(image_name).name)

    return hard_names


def count_hard_examples_in_subset(train_dataset: Subset, hard_example_names: set[str]) -> int:
    if not hard_example_names:
        return 0

    base_dataset = train_dataset.dataset
    indices = list(train_dataset.indices)
    return sum(1 for index in indices if base_dataset.samples[index][0].name in hard_example_names)


class TrainEvalSubsetDataset(Dataset):
    def __init__(self, train_subset: Subset) -> None:
        self.train_subset = train_subset
        self.base_dataset = train_subset.dataset
        self.indices = list(train_subset.indices)
        self.eval_transform = make_eval_transform()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        base_index = self.indices[index]
        image_path, label = self.base_dataset.samples[base_index]
        with Image.open(image_path) as image:
            tensor = self.eval_transform(image)
        return tensor, int(label), image_path.name


def mine_hard_examples_from_train(
    model: nn.Module,
    train_dataset: Subset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
    log_interval: int,
    start_time: float,
) -> set[str]:
    eval_subset = TrainEvalSubsetDataset(train_dataset)
    eval_loader = make_loader(
        eval_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        persistent_workers=False,
    )
    total_batches = len(eval_loader)
    log_message(start_time, f"  Hard mining batches: {total_batches} | workers: {max(0, int(num_workers))}")

    hard_names: set[str] = set()
    model.eval()

    with torch.no_grad():
        for batch_index, (images, labels, names) in enumerate(eval_loader, start=1):
            images = images.to(device)
            labels = labels.long().to(device)

            logits = model(images)
            probs = torch.sigmoid(logits).view(-1)
            predictions = (probs >= threshold).long()
            mismatches = predictions != labels

            for index, is_mismatch in enumerate(mismatches.tolist()):
                if is_mismatch:
                    hard_names.add(str(names[index]))

            if batch_index == 1 or batch_index % max(1, log_interval) == 0 or batch_index == total_batches:
                progress_pct = 100.0 * batch_index / max(1, total_batches)
                log_message(start_time, f"  Hard batch {batch_index}/{total_batches} ({progress_pct:5.1f}%)")

    return hard_names


def build_balanced_train_loader(
    train_dataset: Subset,
    batch_size: int,
    num_workers: int,
    hard_example_names: set[str] | None = None,
    hard_weight_multiplier: float = 1.0,
) -> tuple[DataLoader, WeightedRandomSampler]:
    base_dataset = train_dataset.dataset
    indices = list(train_dataset.indices)
    targets = [int(base_dataset.targets[i]) for i in indices]
    hard_example_names = hard_example_names or set()
    multiplier = max(1.0, float(hard_weight_multiplier))

    class_counts: dict[int, int] = {}
    for target in targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    sample_weights: list[float] = []
    for index, target in zip(indices, targets):
        weight = 1.0 / class_counts[target]
        image_name = base_dataset.samples[index][0].name
        if image_name in hard_example_names:
            weight *= multiplier
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    worker_count = max(0, int(num_workers))
    if worker_count > 0:
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )
        return loader, sampler

    loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())
    return loader, sampler


def update_balanced_sampler_weights(
    train_dataset: Subset,
    sampler: WeightedRandomSampler,
    hard_example_names: set[str] | None = None,
    hard_weight_multiplier: float = 1.0,
) -> None:
    base_dataset = train_dataset.dataset
    indices = list(train_dataset.indices)
    targets = [int(base_dataset.targets[i]) for i in indices]
    hard_example_names = hard_example_names or set()
    multiplier = max(1.0, float(hard_weight_multiplier))

    class_counts: dict[int, int] = {}
    for target in targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    sample_weights: list[float] = []
    for index, target in zip(indices, targets):
        weight = 1.0 / class_counts[target]
        image_name = base_dataset.samples[index][0].name
        if image_name in hard_example_names:
            weight *= multiplier
        sample_weights.append(weight)

    sampler.weights = torch.tensor(sample_weights, dtype=torch.double)


def save_logs(rows: list[dict[str, float]]) -> None:
    with LOG_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_tp",
                "train_tn",
                "train_fp",
                "train_fn",
                "val_loss",
                "val_accuracy",
                "val_precision",
                "val_recall",
                "val_specificity",
                "val_f1",
                "val_balanced_accuracy",
                "val_selection_score",
                "val_tp",
                "val_tn",
                "val_fp",
                "val_fn",
                "val_threshold",
                "elapsed_seconds",
                "elapsed_time",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_threshold(threshold: float) -> None:
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with THRESHOLD_PATH.open("w", encoding="utf-8") as file:
        json.dump({"threshold": threshold}, file, indent=2)


def main() -> None:
    args = parse_args()
    start_time = perf_counter()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_output_dirs()

    if not 0.0 <= args.dynamic_hard_threshold <= 1.0:
        raise ValueError("--dynamic-hard-threshold must be in [0.0, 1.0].")

    log_message(start_time, "Loading datasets...")
    split_data, class_weights = load_datasets(args.data_root, args.seed, args.val)

    static_hard_example_names = collect_hard_example_names(args.hard_examples_dir, args.hard_examples_csv)
    dynamic_hard_example_names: set[str] = set()
    effective_multiplier = max(1.0, float(args.hard_weight_multiplier))

    combined_hard_names = static_hard_example_names | dynamic_hard_example_names
    train_hard_count = count_hard_examples_in_subset(split_data.train_dataset, combined_hard_names)

    train_loader, train_sampler = build_balanced_train_loader(
        split_data.train_dataset,
        args.batch_size,
        args.num_workers,
        hard_example_names=combined_hard_names,
        hard_weight_multiplier=effective_multiplier,
    )
    val_loader = make_loader(split_data.val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    log_message(start_time, "Datasets loaded.")
    log_message(start_time, f"Classes: {split_data.class_names}")
    log_message(start_time, f"Device: {DEVICE}")
    log_message(start_time, f"DataLoader workers: {args.num_workers}")
    log_message(start_time, f"Train samples: {len(split_data.train_dataset)}")
    log_message(start_time, f"Val samples: {len(split_data.val_dataset)}")
    log_message(start_time, f"Hard examples found from inputs (by name): {len(static_hard_example_names)}")
    log_message(start_time, f"Dynamic hard mining enabled: {args.dynamic_hard_mining}")
    log_message(start_time, f"Dynamic hard mining threshold: {args.dynamic_hard_threshold:.2f}")
    log_message(start_time, f"Dynamic hard mining workers: {max(0, int(args.hard_mining_workers))}")
    log_message(start_time, f"Hard examples present in train split: {train_hard_count}")
    log_message(start_time, f"Hard sampling multiplier: {effective_multiplier:.2f}")

    model = CNN().to(DEVICE)

    raw_pos_weight = (class_weights[1] / class_weights[0]).item() if len(class_weights) > 1 else 1.0
    capped_pos_weight = min(max(raw_pos_weight, args.min_pos_weight), args.max_pos_weight)
    pos_weight = torch.tensor(capped_pos_weight, dtype=torch.float32, device=DEVICE)
    criterion = FocalBCEWithLogitsLoss(gamma=args.focal_gamma, alpha=args.focal_alpha, pos_weight=pos_weight).to(DEVICE)

    log_message(start_time, f"Class weights: {class_weights.cpu().tolist()}")
    log_message(start_time, f"Raw pos_weight: {raw_pos_weight:.4f}")
    log_message(start_time, f"Capped pos_weight: {capped_pos_weight:.4f}")
    log_message(start_time, f"Focal loss gamma: {args.focal_gamma:.2f}")
    log_message(start_time, f"Focal loss alpha: {args.focal_alpha:.2f}")
    log_message(start_time, f"RGB auxiliary loss weight: {args.rgb_loss_weight:.2f}")
    log_message(start_time, f"Residual auxiliary loss weight: {args.residual_loss_weight:.2f}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_warmup_cosine_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_ratio=args.min_lr_ratio,
    )
    ema = ExponentialMovingAverage(model, decay=args.ema_decay)
    log_message(start_time, f"EMA decay: {args.ema_decay:.4f}")
    log_message(start_time, f"Warmup epochs: {args.warmup_epochs}")
    log_message(start_time, f"Min LR ratio: {args.min_lr_ratio:.2f}")

    best_val_selection_score = -1.0
    best_epoch = 0
    best_threshold = 0.5
    epochs_without_improvement = 0
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        combined_hard_names = static_hard_example_names | dynamic_hard_example_names
        train_hard_count = count_hard_examples_in_subset(split_data.train_dataset, combined_hard_names)
        update_balanced_sampler_weights(
            split_data.train_dataset,
            train_sampler,
            hard_example_names=combined_hard_names,
            hard_weight_multiplier=effective_multiplier,
        )

        log_message(start_time, f"Epoch [{epoch}/{args.epochs}] started")
        log_message(
            start_time,
            f"  Sampler hard examples in train split: {train_hard_count} (static={len(static_hard_example_names)}, dynamic={len(dynamic_hard_example_names)})",
        )

        train_scores = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            args.log_interval,
            start_time,
            ema=ema,
            rgb_loss_weight=args.rgb_loss_weight,
            residual_loss_weight=args.residual_loss_weight,
        )

        ema.apply(model)
        val_scores, val_threshold = evaluate_with_threshold_search(
            model,
            val_loader,
            criterion,
            DEVICE,
            args.log_interval,
            start_time,
            rgb_loss_weight=args.rgb_loss_weight,
            residual_loss_weight=args.residual_loss_weight,
        )
        ema.restore(model)

        current_lr = optimizer.param_groups[0]["lr"]

        log_message(start_time, f"Epoch [{epoch}/{args.epochs}] finished")
        log_message(
            start_time,
            f"Train loss: {train_scores['loss']:.4f} acc: {train_scores['accuracy']:.4f} "
            f"| Val loss: {val_scores['loss']:.4f} acc: {val_scores['accuracy']:.4f} "
            f"bal_acc: {val_scores['balanced_accuracy']:.4f} score: {val_scores['selection_score']:.4f}",
        )
        log_message(
            start_time,
            f"Train CM TP/TN/FP/FN: {int(train_scores['tp'])}/{int(train_scores['tn'])}/{int(train_scores['fp'])}/{int(train_scores['fn'])}",
        )
        log_message(
            start_time,
            f"Val   CM TP/TN/FP/FN: {int(val_scores['tp'])}/{int(val_scores['tn'])}/{int(val_scores['fp'])}/{int(val_scores['fn'])} | thr={val_threshold:.2f}",
        )
        log_message(start_time, f"LR: {current_lr:.6f}")

        if args.dynamic_hard_mining:
            log_message(start_time, "Starting dynamic hard example mining on train split...")
            dynamic_hard_example_names = mine_hard_examples_from_train(
                model,
                split_data.train_dataset,
                args.batch_size,
                args.hard_mining_workers,
                DEVICE,
                args.dynamic_hard_threshold,
                args.log_interval,
                start_time,
            )
            log_message(start_time, f"Dynamic hard examples mined from train split: {len(dynamic_hard_example_names)}")

        log_message(start_time, "-" * 60)

        epoch_elapsed_seconds = perf_counter() - start_time
        log_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_scores["loss"],
                "train_accuracy": train_scores["accuracy"],
                "train_tp": train_scores["tp"],
                "train_tn": train_scores["tn"],
                "train_fp": train_scores["fp"],
                "train_fn": train_scores["fn"],
                "val_loss": val_scores["loss"],
                "val_accuracy": val_scores["accuracy"],
                "val_precision": val_scores["precision"],
                "val_recall": val_scores["recall"],
                "val_specificity": val_scores["specificity"],
                "val_f1": val_scores["f1"],
                "val_balanced_accuracy": val_scores["balanced_accuracy"],
                "val_selection_score": val_scores["selection_score"],
                "val_tp": val_scores["tp"],
                "val_tn": val_scores["tn"],
                "val_fp": val_scores["fp"],
                "val_fn": val_scores["fn"],
                "val_threshold": val_threshold,
                "elapsed_seconds": round(epoch_elapsed_seconds, 3),
                "elapsed_time": format_elapsed(epoch_elapsed_seconds),
            }
        )

        improvement = val_scores["selection_score"] - best_val_selection_score
        if improvement > args.early_stop_min_delta:
            best_val_selection_score = val_scores["selection_score"]
            best_epoch = epoch
            best_threshold = val_threshold
            epochs_without_improvement = 0

            ema.apply(model)
            torch.save({"state_dict": model.state_dict(), "threshold": best_threshold}, MODEL_PATH)
            ema.restore(model)
            save_threshold(best_threshold)
            log_message(start_time, f"Model saved to {MODEL_PATH}")
            log_message(start_time, f"Threshold saved to {THRESHOLD_PATH}")
        else:
            epochs_without_improvement += 1

        scheduler.step()

        if epochs_without_improvement >= args.early_stop_patience:
            log_message(
                start_time,
                f"Early stopping triggered at epoch {epoch}: "
                f"no val_selection_score improvement > {args.early_stop_min_delta:.4f} "
                f"for {args.early_stop_patience} epochs.",
            )
            break

    save_logs(log_rows)
    save_training_plots(log_rows, PLOT_PATH, start_time)
    log_message(start_time, f"Training finished. Best val selection score: {best_val_selection_score:.4f}")
    log_message(start_time, f"Best epoch: {best_epoch}")
    log_message(start_time, f"Best validation threshold: {best_threshold:.2f}")
    log_message(start_time, f"Logs saved to {LOG_PATH}")
    log_message(start_time, f"Training diagnostics plot saved to {PLOT_PATH}")


if __name__ == "__main__":
    main()

# python .\src\v8\train.py --epochs 50 --hard-weight-multiplier 1.0 --focal-gamma 1.0 --focal-alpha 0.5 --ema-decay 0.995 --min-lr-ratio 0.2
