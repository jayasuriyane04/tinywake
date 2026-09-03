"""
training/evaluate.py

Evaluates the trained model (kws_fp32.keras by default, or pass
--model to point at the INT8 .tflite) on the held-out test split:
  - accuracy / precision / recall / confusion matrix
  - true positive rate & false rejection rate for the wake word
  - false positive rate (non-wake classified as wake)
  - a confidence-threshold sweep to recommend WAKE_THRESHOLD

Usage:
    python training/evaluate.py
    python training/evaluate.py --model models/kws_int8.tflite
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from training.preprocess import preprocess_waveform  # noqa: E402
from training.features import waveform_to_mfcc_numpy  # noqa: E402

log = get_logger("EVALUATE")

LABEL_TO_INDEX = {label: i for i, label in enumerate(config.LABELS)}
WAKE_INDEX = LABEL_TO_INDEX[config.WAKE_WORD]


def _load_test_entries() -> list[dict]:
    manifest_path = config.DATASET_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest["test"] or manifest["val"]


def _predict_probs_keras(model_path: Path, features: np.ndarray) -> np.ndarray:
    model = tf.keras.models.load_model(model_path)
    return model.predict(features, verbose=0)


def _predict_probs_tflite(model_path: Path, features: np.ndarray) -> np.ndarray:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    in_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    in_scale, in_zero = in_detail["quantization"]
    out_scale, out_zero = out_detail["quantization"]

    probs = np.zeros((len(features), len(config.LABELS)), dtype=np.float32)
    for i, feat in enumerate(features):
        x = feat[np.newaxis, ...]
        if in_detail["dtype"] == np.int8:
            x = (x / in_scale + in_zero).round().astype(np.int8)
        interpreter.set_tensor(in_detail["index"], x)
        interpreter.invoke()
        out = interpreter.get_tensor(out_detail["index"])[0]
        if out_detail["dtype"] == np.int8:
            out = (out.astype(np.float32) - out_zero) * out_scale
        probs[i] = out
    return probs


def evaluate(model_path: Path) -> dict:
    entries = _load_test_entries()
    log.info(f"evaluating on {len(entries)} test samples")

    features_list, labels_list = [], []
    for e in entries:
        waveform = preprocess_waveform(e["path"])
        feats = waveform_to_mfcc_numpy(waveform)
        features_list.append(feats)
        labels_list.append(LABEL_TO_INDEX[e["label"]])

    features = np.stack(features_list)
    labels = np.array(labels_list)

    if model_path.suffix == ".tflite":
        probs = _predict_probs_tflite(model_path, features)
    else:
        probs = _predict_probs_keras(model_path, features)

    preds = np.argmax(probs, axis=1)

    # Confusion matrix
    num_classes = len(config.LABELS)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1

    accuracy = float(np.mean(preds == labels))

    # Wake-word-specific stats at argmax (no threshold)
    is_wake_true = labels == WAKE_INDEX
    is_wake_pred = preds == WAKE_INDEX
    tp = int(np.sum(is_wake_true & is_wake_pred))
    fn = int(np.sum(is_wake_true & ~is_wake_pred))
    fp = int(np.sum(~is_wake_true & is_wake_pred))
    tn = int(np.sum(~is_wake_true & ~is_wake_pred))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    tpr = recall
    frr = fn / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    # Threshold sweep on P(hey_nova)
    wake_probs = probs[:, WAKE_INDEX]
    sweep = []
    for thresh in config.CANDIDATE_THRESHOLDS:
        pred_wake = wake_probs >= thresh
        t_tp = int(np.sum(is_wake_true & pred_wake))
        t_fn = int(np.sum(is_wake_true & ~pred_wake))
        t_fp = int(np.sum(~is_wake_true & pred_wake))
        t_tn = int(np.sum(~is_wake_true & ~pred_wake))
        t_tpr = t_tp / (t_tp + t_fn) if (t_tp + t_fn) else 0.0
        t_fpr = t_fp / (t_fp + t_tn) if (t_fp + t_tn) else 0.0
        # Youden's J statistic to rank tradeoffs
        j_stat = t_tpr - t_fpr
        sweep.append({
            "threshold": thresh, "tpr": t_tpr, "fpr": t_fpr,
            "false_positives": t_fp, "j_stat": j_stat,
        })

    best = max(sweep, key=lambda r: r["j_stat"])

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_positive_rate": tpr,
        "false_rejection_rate": frr,
        "false_positive_rate": fpr,
        "confusion_matrix": cm.tolist(),
        "labels": config.LABELS,
        "threshold_sweep": sweep,
        "recommended_threshold": best["threshold"],
        "num_test_samples": len(entries),
    }

    log.info(f"accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f}")
    log.info(f"wake-word TPR={tpr:.3f} FRR={frr:.3f} FPR={fpr:.3f}")
    log.info(f"confusion matrix (rows=true, cols=pred, labels={config.LABELS}):\n{cm}")
    log.info("threshold sweep:")
    for r in sweep:
        marker = "  <-- recommended" if r["threshold"] == best["threshold"] else ""
        log.info(f"  t={r['threshold']:.2f}  TPR={r['tpr']:.3f}  FPR={r['fpr']:.3f}  "
                  f"FP={r['false_positives']}{marker}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(config.MODELS_DIR / "kws_fp32.keras"))
    parser.add_argument("--out", default=str(config.MODELS_DIR / "evaluation_report.json"))
    args = parser.parse_args()

    results = evaluate(Path(args.model))
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"report saved to {args.out}")


if __name__ == "__main__":
    main()
