"""
training/quantize.py

Converts the trained Keras model (models/kws_fp32.keras) into a
fully INT8-quantized TFLite model (weights, activations, input, and
output all INT8) using a representative dataset drawn from the
training split. Output is compatible with TensorFlow Lite Micro.

Usage:
    python training/quantize.py
"""
from __future__ import annotations
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
from training.evaluate import evaluate  # noqa: E402

log = get_logger("QUANTIZE")

NUM_CALIBRATION_SAMPLES = 200


def _load_manifest_paths() -> list[str]:
    manifest_path = config.DATASET_DIR / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)
    entries = manifest["train"] or manifest["val"] or manifest["test"]
    return [e["path"] for e in entries]


def _representative_dataset_gen():
    paths = _load_manifest_paths()
    rng = np.random.RandomState(0)
    rng.shuffle(paths)
    for path in paths[:NUM_CALIBRATION_SAMPLES]:
        waveform = preprocess_waveform(path)
        feats = waveform_to_mfcc_numpy(waveform)
        yield [feats[np.newaxis, ...].astype(np.float32)]


def quantize(keras_model_path: Path, out_path: Path) -> None:
    model = tf.keras.models.load_model(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = _representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_model)

    fp32_size = keras_model_path.stat().st_size if keras_model_path.is_file() else _dir_size(keras_model_path)
    int8_size = out_path.stat().st_size
    ratio = fp32_size / int8_size if int8_size else float("inf")

    log.info(f"FP32 model size: {fp32_size / 1024:.1f} KB")
    log.info(f"INT8 model size: {int8_size / 1024:.1f} KB")
    log.info(f"compression ratio: {ratio:.2f}x")

    if int8_size > 100 * 1024:
        log.info("WARNING: INT8 model exceeds the 100KB target — consider shrinking model.py")


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main() -> None:
    keras_path = config.MODELS_DIR / "kws_fp32.keras"
    int8_path = config.MODELS_DIR / "kws_int8.tflite"

    if not keras_path.exists():
        raise FileNotFoundError(f"{keras_path} not found — run training/train.py first.")

    log.info("evaluating FP32 model accuracy before quantization...")
    fp32_results = evaluate(keras_path)

    log.info("quantizing to INT8...")
    quantize(keras_path, int8_path)

    log.info("evaluating INT8 model accuracy after quantization...")
    int8_results = evaluate(int8_path)

    log.info(f"accuracy before quantization: {fp32_results['accuracy']:.3f}")
    log.info(f"accuracy after quantization:  {int8_results['accuracy']:.3f}")
    delta = int8_results["accuracy"] - fp32_results["accuracy"]
    log.info(f"accuracy delta: {delta:+.3f}")


if __name__ == "__main__":
    main()
