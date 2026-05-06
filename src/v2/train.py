"""Training script for AI Image Detector CNN."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model import CNN
from dataset import load_datasets
from evaluate import evaluate
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
    ensure_output_dirs,
    make_loader,
    compute_scores,
    accumulate_predictions,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train CNN for AI image detection.")
    parser.add_argument("--data-root", type=Path, default=DATASET_ROOT, help="Path to dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE, help="Learning rate for Adam.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="L2 regularization weight decay.")
    parser.add_argument("--val", type=float, default=VAL_FRACTION, help="Validation dataset fraction of training dataset.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducible splitting.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of worker processes.")
    parser.add_argument("--log-interval", type=int, default=200, help="Print progress every N batches.")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    log_interval: int,
) -> dict[str, float]:
    """Train for one epoch."""
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


def save_logs(rows: list[dict[str, float]]) -> None:
    """Save training logs to CSV."""
    log_path = OUTPUT_DIR / "logs" / "training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_precision", "val_recall", "val_f1"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    ensure_output_dirs()

    # Load datasets
    print("Loading datasets...")
    split_data, class_weights = load_datasets(args.data_root, args.seed, args.val)

    train_loader = make_loader(split_data.train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(split_data.val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print("Datasets loaded.")
    print(f"Classes: {split_data.class_names}")
    print(f"Device: {DEVICE}")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"Train samples: {len(split_data.train_dataset)}")
    print(f"Val samples: {len(split_data.val_dataset)}")

    model = CNN().to(DEVICE)

    # Setup loss with class weights
    pos_weight = class_weights[1] / class_weights[0] if len(class_weights) > 1 else torch.tensor(1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))

    print(f"Class weights: {class_weights.cpu().numpy()}")
    print(f"Positive weight: {pos_weight.item():.4f}")

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    best_val_accuracy = 0.0
    best_model_weights: dict[str, torch.Tensor] | None = None
    log_rows: list[dict[str, float]] = []
    model_path = OUTPUT_DIR / "models" / "model2.pth"

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
            torch.save(best_model_weights, model_path)
            print(f"Model saved to {model_path}")

    save_logs(log_rows)
    print(f"Training finished. Best val accuracy: {best_val_accuracy:.4f}")
    print(f"Logs saved to {OUTPUT_DIR / 'logs' / 'training_log.csv'}")


if __name__ == "__main__":
    main()
