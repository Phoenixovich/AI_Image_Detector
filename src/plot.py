from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot metrics from test summary CSV or per-image test results CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more CSV files (test_summary_*.csv or test_results_*.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "plots",
        help="Directory where PNG plots are saved.",
    )
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        headers = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return headers, rows


def as_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def as_int(value: str | None) -> int | None:
    number = as_float(value)
    if number is None:
        return None
    return int(number)


def compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "specificity": specificity,
    }


def is_summary_format(headers: Iterable[str]) -> bool:
    normalized = {h.strip().lower() for h in headers}
    return {"metric", "value"}.issubset(normalized)


def is_results_format(headers: Iterable[str]) -> bool:
    normalized = {h.strip().lower() for h in headers}
    required = {"actual_label", "predicted_label", "confidence"}
    return required.issubset(normalized)


def plot_summary_csv(csv_path: Path, rows: list[dict[str, str]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    metrics: list[str] = []
    values: list[float] = []

    for row in rows:
        metric = (row.get("metric") or "").strip()
        value = as_float(row.get("value"))
        if metric and value is not None:
            metrics.append(metric)
            values.append(value)

    if not metrics:
        raise ValueError(f"No numeric metric/value rows found in {csv_path}")

    fig, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(metrics, values)
    axis.set_title(f"Summary Metrics - {csv_path.name}")
    axis.set_ylabel("Value")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.tick_params(axis="x", rotation=35)

    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = output_dir / f"{csv_path.stem}_metrics_{timestamp}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_results_csv(csv_path: Path, rows: list[dict[str, str]], output_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    tp = tn = fp = fn = 0
    labeled_conf_correct: list[float] = []
    labeled_conf_incorrect: list[float] = []
    total_rows = len(rows)
    labeled_rows = 0

    for row in rows:
        actual = as_int(row.get("actual_label"))
        pred = as_int(row.get("predicted_label"))
        conf = as_float(row.get("confidence"))

        if actual not in (0, 1) or pred not in (0, 1):
            continue

        labeled_rows += 1
        if actual == 1 and pred == 1:
            tp += 1
        elif actual == 0 and pred == 0:
            tn += 1
        elif actual == 0 and pred == 1:
            fp += 1
        elif actual == 1 and pred == 0:
            fn += 1

        if conf is not None:
            if actual == pred:
                labeled_conf_correct.append(conf)
            else:
                labeled_conf_incorrect.append(conf)

    if labeled_rows == 0:
        raise ValueError(f"No labeled rows found in {csv_path}")

    metrics = compute_binary_metrics(tp, tn, fp, fn)
    confusion = np.array([[tn, fp], [fn, tp]], dtype=int)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax_conf = axes[0, 0]
    image = ax_conf.imshow(confusion, cmap="Blues")
    ax_conf.set_title("Confusion Matrix")
    ax_conf.set_xlabel("Predicted")
    ax_conf.set_ylabel("Actual")
    ax_conf.set_xticks([0, 1], labels=["0", "1"])
    ax_conf.set_yticks([0, 1], labels=["0", "1"])
    for i in range(2):
        for j in range(2):
            ax_conf.text(j, i, str(confusion[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax_conf, fraction=0.046, pad=0.04)

    ax_hist = axes[0, 1]
    bins = np.linspace(0, 1, 21)
    ax_hist.hist(labeled_conf_correct, bins=bins, alpha=0.7, label="correct")
    ax_hist.hist(labeled_conf_incorrect, bins=bins, alpha=0.7, label="incorrect")
    ax_hist.set_title("Confidence Distribution")
    ax_hist.set_xlabel("Confidence")
    ax_hist.set_ylabel("Count")
    ax_hist.legend()
    ax_hist.grid(alpha=0.3)

    ax_metrics = axes[1, 0]
    metric_names = ["accuracy", "precision", "recall", "f1", "specificity"]
    metric_values = [metrics[name] for name in metric_names]
    bars = ax_metrics.bar(metric_names, metric_values)
    ax_metrics.set_ylim(0.0, 1.0)
    ax_metrics.set_title("Binary Metrics")
    ax_metrics.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, metric_values):
        ax_metrics.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax_counts = axes[1, 1]
    unlabeled_rows = total_rows - labeled_rows
    count_names = ["labeled", "unlabeled", "TP", "TN", "FP", "FN"]
    count_values = [labeled_rows, unlabeled_rows, tp, tn, fp, fn]
    ax_counts.bar(count_names, count_values)
    ax_counts.set_title("Row/Confusion Counts")
    ax_counts.grid(axis="y", linestyle="--", alpha=0.4)
    ax_counts.tick_params(axis="x", rotation=25)

    fig.suptitle(f"Test Results Analysis - {csv_path.name}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = output_dir / f"{csv_path.stem}_analysis_{timestamp}.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def process_input(csv_path: Path, output_dir: Path) -> list[Path]:
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    headers, rows = read_csv_rows(csv_path)
    if not headers:
        raise ValueError(f"CSV has no headers: {csv_path}")

    if is_summary_format(headers):
        return [plot_summary_csv(csv_path, rows, output_dir)]

    if is_results_format(headers):
        return [plot_results_csv(csv_path, rows, output_dir)]

    raise ValueError(
        "Unsupported CSV format. Expected either: "
        "(metric,value) summary CSV, or per-image CSV with "
        "actual_label,predicted_label,confidence columns."
    )


def main() -> None:
    args = parse_args()

    try:
        import matplotlib  # type: ignore

        matplotlib.use("Agg")
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_outputs: list[Path] = []
    for csv_path in args.input:
        output_files = process_input(csv_path, args.output_dir)
        all_outputs.extend(output_files)

    print("Generated plots:")
    for path in all_outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
