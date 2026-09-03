"""
benchmarks/accuracy_test.py

Runs the full test dataset through the INT8 model and reports
TP/TN/FP/FN, precision/recall/F1, false rejection/acceptance rate,
and saves a confusion matrix image.

Usage:
    python benchmarks/accuracy_test.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from training.evaluate import evaluate, WAKE_INDEX  # noqa: E402

log = get_logger("ACCURACY")


def plot_confusion_matrix(cm: list[list[int]], labels: list[str], out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("tinywake KWS confusion matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    log.info(f"confusion matrix image saved to {out_path}")


def main() -> None:
    model_path = config.MODELS_DIR / "kws_int8.tflite"
    if not model_path.exists():
        model_path = config.MODELS_DIR / "kws_fp32.keras"
    if not model_path.exists():
        log.info("no trained model found — run training/train.py (and quantize.py) first")
        sys.exit(1)

    results = evaluate(model_path)
    cm = results["confusion_matrix"]
    labels = results["labels"]

    tp = cm[WAKE_INDEX][WAKE_INDEX]
    fn = sum(cm[WAKE_INDEX]) - tp
    fp = sum(cm[i][WAKE_INDEX] for i in range(len(labels))) - tp
    tn = sum(sum(row) for row in cm) - tp - fn - fp

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    frr = fn / (tp + fn) if (tp + fn) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0

    log.info(f"TP={tp} TN={tn} FP={fp} FN={fn}")
    log.info(f"precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
    log.info(f"false_rejection_rate={frr:.3f} false_acceptance_rate={far:.3f}")

    out_path = config.BENCHMARK_RESULTS_DIR / "confusion_matrix.png"
    config.BENCHMARK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(cm, labels, out_path)


if __name__ == "__main__":
    main()
