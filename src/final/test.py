from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import CNN
from utils import (
    PROJECT_ROOT,
    OUTPUT_DIR,
    BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEVICE,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    make_eval_transform,
    make_loader,
    compute_scores,
    confusion_from_predictions,
    format_elapsed,
)


DEFAULT_MODEL_PATH = OUTPUT_DIR / "models" / "model_final.pth"
DEFAULT_THRESHOLD_PATH = OUTPUT_DIR / "models" / "model_final_threshold.json"
DEFAULT_TEST_DIR = PROJECT_ROOT / "dataset" / "processed" / "test"
DEFAULT_TEST_LABELS_PATH = PROJECT_ROOT / "dataset" / "raw" / "test_labels.csv"
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "logs" / f"test_results_final_{RUN_TIMESTAMP}.csv"
DEFAULT_SUMMARY_PATH = OUTPUT_DIR / "logs" / f"test_summary_final_{RUN_TIMESTAMP}.csv"
DEFAULT_EXPLANATIONS_DIR = OUTPUT_DIR / "explanations" / f"final_{RUN_TIMESTAMP}"


def normalize_label_key(file_id: str) -> str:
    """Normalize file ids so differently padded names resolve to the same key."""
    stem = Path(file_id.strip()).stem.lower()
    match = re.match(r"^([a-z]+)(\d+)$", stem)
    if match:
        prefix, number = match.groups()
        return f"{prefix}{int(number)}"
    return stem


def read_label_map(test_labels_path: Path) -> dict[str, int]:
    label_map: dict[str, int] = {}
    if not test_labels_path.exists():
        return label_map

    with test_labels_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Support common column names from different export scripts.
            file_id = (
                row.get("id")
                or row.get("image")
                or row.get("filename")
                or row.get("file")
                or ""
            ).strip()
            label_raw = (row.get("label") or row.get("target") or row.get("class") or "").strip()

            if not file_id or label_raw == "":
                continue

            try:
                label = int(float(label_raw))
            except ValueError:
                continue

            label_map[normalize_label_key(file_id)] = label

    return label_map


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

        label_map = read_label_map(test_labels_path)

        for img_path in sorted(test_dir.glob("*.*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label = label_map.get(normalize_label_key(img_path.name))

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
    parser = argparse.ArgumentParser(description="Evaluate final gated multi-evidence CNN model on test set.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to trained final model.")
    parser.add_argument("--threshold-path", type=Path, default=DEFAULT_THRESHOLD_PATH, help="Path to threshold JSON saved by training.")
    parser.add_argument("--threshold", type=float, default=None, help="Override threshold (if provided, ignores threshold file).")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR, help="Directory with test images.")
    parser.add_argument("--labels-path", type=Path, default=DEFAULT_TEST_LABELS_PATH, help="CSV file with test labels.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for testing.")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS, help="Number of DataLoader workers.")
    parser.add_argument("--log-interval", type=int, default=25, help="Print progress every N batches.")
    parser.add_argument("--results-path", type=Path, default=DEFAULT_RESULTS_PATH, help="CSV output path for per-image results.")
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH, help="CSV output path for summary metrics.")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation (flip + scale views).")
    parser.add_argument("--explain-regions", action="store_true", help="Save Grad-CAM heatmaps showing regions that influenced each prediction.")
    parser.add_argument("--explanations-dir", type=Path, default=DEFAULT_EXPLANATIONS_DIR, help="Directory for Grad-CAM explanation images.")
    parser.add_argument("--explain-limit", type=int, default=100, help="Maximum number of images to explain. Use 0 for all images.")
    return parser.parse_args()


def tta_logits(model: nn.Module, images: torch.Tensor, use_tta: bool) -> torch.Tensor:
    logits: list[torch.Tensor] = [model(images)]
    if not use_tta:
        return logits[0]

    logits.append(model(torch.flip(images, dims=[3])))

    upscaled = F.interpolate(images, scale_factor=1.10, mode="bilinear", align_corners=False)
    upscaled = F.interpolate(upscaled, size=images.shape[-2:], mode="bilinear", align_corners=False)
    logits.append(model(upscaled))

    return torch.stack(logits, dim=0).mean(dim=0)


def log_message(start_time: float, message: str) -> None:
    elapsed = format_elapsed(perf_counter() - start_time)
    print(f"[{elapsed}] {message}", flush=True)


def area_name(x: int, y: int, width: int, height: int) -> str:
    horizontal = "left" if x < width / 3 else "right" if x >= 2 * width / 3 else "center"
    vertical = "top" if y < height / 3 else "bottom" if y >= 2 * height / 3 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "middle":
        return horizontal
    return f"{vertical}-{horizontal}"


def tensor_to_pil_image(image: torch.Tensor) -> Image.Image:
    mean = NORMALIZE_MEAN.to(image.device)
    std = NORMALIZE_STD.to(image.device)
    denormalized = (image.detach() * std + mean).clamp(0.0, 1.0)
    array = (denormalized.cpu().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def make_heatmap_overlay(image: torch.Tensor, heatmap: torch.Tensor) -> Image.Image:
    base = tensor_to_pil_image(image).convert("RGBA")
    heatmap = heatmap.detach().cpu().clamp(0.0, 1.0)
    alpha = (heatmap * 150.0).byte().numpy()
    red = torch.full_like(heatmap, 255, dtype=torch.uint8).numpy()
    green = (heatmap * 80.0).byte().numpy()
    blue = torch.zeros_like(heatmap, dtype=torch.uint8).numpy()
    overlay = Image.merge(
        "RGBA",
        (
            Image.fromarray(red, mode="L"),
            Image.fromarray(green, mode="L"),
            Image.fromarray(blue, mode="L"),
            Image.fromarray(alpha, mode="L"),
        ),
    )
    return Image.alpha_composite(base, overlay).convert("RGB")


def gradcam_explanation(
    model: CNN,
    image: torch.Tensor,
    prediction: int,
    image_name: str,
    explanations_dir: Path,
) -> dict[str, str]:
    target_layer = model.rgb_branch.features[-3]
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)

    def backward_hook(_module: nn.Module, _grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]) -> None:
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        input_image = image.unsqueeze(0).to(next(model.parameters()).device)
        logits = model(input_image)
        target = logits.view(-1)[0] if prediction == 1 else -logits.view(-1)[0]
        target.backward()

        if not activations or not gradients:
            return {"decision_region": "", "region_x": "", "region_y": "", "heatmap_path": ""}

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[0, 0]
        cam_min = cam.min()
        cam_max = cam.max()
        heatmap = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        flat_index = int(torch.argmax(heatmap).item())
        y, x = divmod(flat_index, heatmap.shape[1])
        region = area_name(x, y, heatmap.shape[1], heatmap.shape[0])

        explanations_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(image_name).stem)
        heatmap_path = explanations_dir / f"{safe_stem}_gradcam.jpg"
        make_heatmap_overlay(image, heatmap).save(heatmap_path, quality=95)

        return {
            "decision_region": region,
            "region_x": str(x),
            "region_y": str(y),
            "heatmap_path": str(heatmap_path),
        }
    finally:
        forward_handle.remove()
        backward_handle.remove()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    log_interval: int,
    use_tta: bool,
    start_time: float,
    explain_regions: bool,
    explanations_dir: Path,
    explain_limit: int,
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
            batch_elapsed_seconds = perf_counter() - start_time
            batch_elapsed_time = format_elapsed(batch_elapsed_seconds)

            logits = tta_logits(model, images, use_tta=use_tta)
            probabilities = torch.sigmoid(logits).view(-1)
            predictions = (probabilities >= threshold).long()

            for item_index, (image_name, label_value, prediction, probability) in enumerate(zip(
                image_names,
                labels.detach().cpu().tolist(),
                predictions.detach().cpu().tolist(),
                probabilities.detach().cpu().tolist(),
            )):
                is_labeled = label_value in (0, 1)

                actual_outcome = "ai" if label_value == 1 else "real" if label_value == 0 else ""
                predicted_outcome = "ai" if prediction == 1 else "real"
                explanation = {"decision_region": "", "region_x": "", "region_y": "", "heatmap_path": ""}
                if explain_regions and (explain_limit <= 0 or len(rows) < explain_limit):
                    with torch.enable_grad():
                        explanation = gradcam_explanation(
                            model,
                            images[item_index].detach().cpu(),
                            int(prediction),
                            str(image_name),
                            explanations_dir,
                        )

                rows.append(
                    {
                        "image": image_name,
                        "actual_label": "" if not is_labeled else str(label_value),
                        "predicted_label": str(prediction),
                        "actual_outcome": actual_outcome,
                        "predicted_outcome": predicted_outcome,
                        "result": "" if not is_labeled else ("correct" if label_value == prediction else "incorrect"),
                        "confidence": f"{probability:.4f}",
                        "decision_region": explanation["decision_region"],
                        "region_x": explanation["region_x"],
                        "region_y": explanation["region_y"],
                        "heatmap_path": explanation["heatmap_path"],
                        "elapsed_seconds": f"{batch_elapsed_seconds:.3f}",
                        "elapsed_time": batch_elapsed_time,
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
                labeled_preds = (labeled_probs >= threshold).long()
                labeled_targets = labeled_labels.long().view(-1)

                batch_tp, batch_tn, batch_fp, batch_fn = confusion_from_predictions(labeled_preds, labeled_targets)
                tp += batch_tp
                tn += batch_tn
                fp += batch_fp
                fn += batch_fn

            if batch_index == 1 or batch_index % log_interval == 0 or batch_index == total_batches:
                avg_loss = running_loss / labeled_count if labeled_count else 0.0
                progress_pct = 100.0 * batch_index / max(1, total_batches)
                log_message(start_time, f"  Test batch {batch_index}/{total_batches} ({progress_pct:5.1f}%) | avg loss: {avg_loss:.4f}")

    scores = compute_scores(tp, tn, fp, fn)
    scores["loss"] = running_loss / labeled_count if labeled_count else 0.0
    scores["labeled_samples"] = float(labeled_count)
    scores["total_samples"] = float(len(rows))
    scores["threshold"] = threshold
    scores["elapsed_seconds"] = perf_counter() - start_time
    return scores, rows


def load_threshold(threshold_path: Path) -> float:
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        value = data.get("threshold", 0.5)
        return float(value)
    return 0.5


def save_results(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "actual_label",
                "predicted_label",
                "actual_outcome",
                "predicted_outcome",
                "result",
                "confidence",
                "decision_region",
                "region_x",
                "region_y",
                "heatmap_path",
                "elapsed_seconds",
                "elapsed_time",
            ],
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
    start_time = perf_counter()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {args.test_dir}")

    transform = make_eval_transform()
    test_dataset = TestDataset(args.test_dir, args.labels_path, transform=transform)
    if len(test_dataset) == 0:
        raise ValueError(f"No valid test images found in {args.test_dir}")

    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    log_message(start_time, f"Test samples: {len(test_dataset)}")
    log_message(start_time, f"Device: {DEVICE}")
    log_message(start_time, f"DataLoader workers: {args.num_workers}")

    model = CNN().to(DEVICE)
    checkpoint = torch.load(args.model_path, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
    else:
        model.load_state_dict(checkpoint)
        checkpoint_threshold = 0.5

    threshold = args.threshold if args.threshold is not None else load_threshold(args.threshold_path)
    if args.threshold is None and checkpoint_threshold != 0.5 and not args.threshold_path.exists():
        threshold = checkpoint_threshold

    log_message(start_time, "Running final test evaluation...")
    log_message(start_time, f"Using threshold: {threshold:.2f}")
    log_message(start_time, f"TTA enabled: {args.tta}")
    log_message(start_time, f"Region explanations enabled: {args.explain_regions}")
    if args.explain_regions:
        explain_count = "all images" if args.explain_limit <= 0 else f"first {args.explain_limit} images"
        log_message(start_time, f"Saving Grad-CAM explanations for {explain_count} to {args.explanations_dir}")
    scores, rows = evaluate(
        model,
        test_loader,
        DEVICE,
        threshold,
        args.log_interval,
        use_tta=args.tta,
        start_time=start_time,
        explain_regions=args.explain_regions,
        explanations_dir=args.explanations_dir,
        explain_limit=args.explain_limit,
    )

    log_message(start_time, "=" * 60)
    log_message(start_time, "Final Test Results:")
    log_message(start_time, f"Labeled Samples: {int(scores['labeled_samples'])}/{int(scores['total_samples'])}")
    log_message(start_time, f"Test Loss: {scores['loss']:.4f}")
    log_message(start_time, f"Test Accuracy: {scores['accuracy']:.4f}")
    log_message(start_time, f"Test Precision: {scores['precision']:.4f}")
    log_message(start_time, f"Test Recall: {scores['recall']:.4f}")
    log_message(start_time, f"Test F1: {scores['f1']:.4f}")
    log_message(start_time, f"Threshold: {scores['threshold']:.2f}")
    log_message(start_time, "=" * 60)

    save_results(args.results_path, rows)
    save_summary(args.summary_path, scores)

    log_message(start_time, f"Per-image results saved to {args.results_path}")
    log_message(start_time, f"Summary metrics saved to {args.summary_path}")


if __name__ == "__main__":
    main()
