from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

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
    SEED,
    DEFAULT_NUM_WORKERS,
    make_loader,
    ensure_output_dirs,
    compute_scores,
    confusion_from_predictions,
)

MODEL_PATH = OUTPUT_DIR / "models" / "model3.pth"
THRESHOLD_PATH = OUTPUT_DIR / "models" / "model3_threshold.json"
LOG_PATH = OUTPUT_DIR / "logs" / "training_log_v3.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v3 CNN with balanced sampling and threshold tuning.")
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
    return parser.parse_args()
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
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
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

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
            print(f"  Train batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}", flush=True)

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

            logits = model(images)
            loss = criterion(logits, labels)

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
                print(f"  Val   batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}", flush=True)

    probs = torch.cat(all_probs) if all_probs else torch.empty(0)
    targets = torch.cat(all_targets) if all_targets else torch.empty(0, dtype=torch.long)

    best_threshold = 0.5
    best_balanced_acc = -1.0
    best_conf = (0, 0, 0, 0)

    for step in range(5, 100):
        threshold = step / 100.0
        preds = (probs >= threshold).long()
        tp, tn, fp, fn = confusion_from_predictions(preds, targets)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        balanced_acc = 0.5 * (tpr + tnr)

        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            best_threshold = threshold
            best_conf = (tp, tn, fp, fn)

    tp, tn, fp, fn = best_conf
    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / sample_count if sample_count else 0.0
    scores["balanced_accuracy"] = best_balanced_acc
    scores["tp"] = float(tp)
    scores["tn"] = float(tn)
    scores["fp"] = float(fp)
    scores["fn"] = float(fn)

    return scores, best_threshold


def build_balanced_train_loader(train_dataset: Subset, batch_size: int, num_workers: int) -> DataLoader:
    base_dataset = train_dataset.dataset
    indices = list(train_dataset.indices)
    targets = [int(base_dataset.targets[i]) for i in indices]

    class_counts: dict[int, int] = {}
    for target in targets:
        class_counts[target] = class_counts.get(target, 0) + 1

    sample_weights = [1.0 / class_counts[target] for target in targets]
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    worker_count = max(0, int(num_workers))
    if worker_count > 0:
        return DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=worker_count,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True,
            prefetch_factor=2,
        )

    return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=torch.cuda.is_available())


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
                "val_tp",
                "val_tn",
                "val_fp",
                "val_fn",
                "val_threshold",
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_output_dirs()

    print("Loading datasets...")
    split_data, class_weights = load_datasets(args.data_root, args.seed, args.val)

    train_loader = build_balanced_train_loader(split_data.train_dataset, args.batch_size, args.num_workers)
    val_loader = make_loader(split_data.val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Datasets loaded.")
    print(f"Classes: {split_data.class_names}")
    print(f"Device: {DEVICE}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Train samples: {len(split_data.train_dataset)}")
    print(f"Val samples: {len(split_data.val_dataset)}")

    model = CNN().to(DEVICE)

    raw_pos_weight = (class_weights[1] / class_weights[0]).item() if len(class_weights) > 1 else 1.0
    capped_pos_weight = min(max(raw_pos_weight, args.min_pos_weight), args.max_pos_weight)
    pos_weight = torch.tensor(capped_pos_weight, dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"Raw pos_weight: {raw_pos_weight:.4f}")
    print(f"Capped pos_weight: {capped_pos_weight:.4f}")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_val_balanced_accuracy = -1.0
    best_threshold = 0.5
    log_rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}] started", flush=True)

        train_scores = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, args.log_interval)
        val_scores, val_threshold = evaluate_with_threshold_search(model, val_loader, criterion, DEVICE, args.log_interval)

        print(f"Epoch [{epoch}/{args.epochs}] finished")
        print(
            f"Train loss: {train_scores['loss']:.4f} acc: {train_scores['accuracy']:.4f} "
            f"| Val loss: {val_scores['loss']:.4f} acc: {val_scores['accuracy']:.4f} bal_acc: {val_scores['balanced_accuracy']:.4f}"
        )
        print(
            f"Train CM TP/TN/FP/FN: {int(train_scores['tp'])}/{int(train_scores['tn'])}/{int(train_scores['fp'])}/{int(train_scores['fn'])}"
        )
        print(
            f"Val   CM TP/TN/FP/FN: {int(val_scores['tp'])}/{int(val_scores['tn'])}/{int(val_scores['fp'])}/{int(val_scores['fn'])} | thr={val_threshold:.2f}"
        )
        print("-" * 60)

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
                "val_tp": val_scores["tp"],
                "val_tn": val_scores["tn"],
                "val_fp": val_scores["fp"],
                "val_fn": val_scores["fn"],
                "val_threshold": val_threshold,
            }
        )

        if val_scores["balanced_accuracy"] > best_val_balanced_accuracy:
            best_val_balanced_accuracy = val_scores["balanced_accuracy"]
            best_threshold = val_threshold
            torch.save({"state_dict": model.state_dict(), "threshold": best_threshold}, MODEL_PATH)
            save_threshold(best_threshold)
            print(f"Model saved to {MODEL_PATH}")
            print(f"Threshold saved to {THRESHOLD_PATH}")

    save_logs(log_rows)
    print(f"Training finished. Best val balanced accuracy: {best_val_balanced_accuracy:.4f}")
    print(f"Best validation threshold: {best_threshold:.2f}")
    print(f"Logs saved to {LOG_PATH}")


if __name__ == "__main__":
    main()
