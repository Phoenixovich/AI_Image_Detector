"""Evaluation utilities for model testing."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils import compute_scores, accumulate_predictions


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, log_interval: int = 200) -> dict[str, float]:
    """Evaluate model on dataset."""
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
