"""
training/live_test.py

Quick sanity-check tool: continuously records ~1s windows from the
microphone and prints the model's class probabilities in real time,
without going through the full edge_simulator pipeline (no ring
buffer, no streaming — just "does the model react to me saying the
wake word").

Usage:
    python training/live_test.py
    python training/live_test.py --model models/kws_int8.tflite
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from training.features import waveform_to_mfcc_numpy  # noqa: E402
from training.preprocess import normalize_amplitude, pad_or_crop  # noqa: E402

log = get_logger("LIVE_TEST")


def _build_predictor(model_path: Path):
    import tensorflow as tf

    if model_path.suffix == ".tflite":
        interpreter = tf.lite.Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        in_detail = interpreter.get_input_details()[0]
        out_detail = interpreter.get_output_details()[0]
        in_scale, in_zero = in_detail["quantization"]
        out_scale, out_zero = out_detail["quantization"]

        def predict(feats: np.ndarray) -> np.ndarray:
            x = feats[np.newaxis, ...]
            if in_detail["dtype"] == np.int8:
                x = (x / in_scale + in_zero).round().astype(np.int8)
            interpreter.set_tensor(in_detail["index"], x)
            interpreter.invoke()
            out = interpreter.get_tensor(out_detail["index"])[0]
            if out_detail["dtype"] == np.int8:
                out = (out.astype(np.float32) - out_zero) * out_scale
            return out

        return predict

    model = tf.keras.models.load_model(model_path)

    def predict(feats: np.ndarray) -> np.ndarray:
        return model.predict(feats[np.newaxis, ...], verbose=0)[0]

    return predict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(config.MODELS_DIR / "kws_fp32.keras"))
    args = parser.parse_args()

    import sounddevice as sd

    predict = _build_predictor(Path(args.model))
    log.info(f"loaded model: {args.model}")
    log.info("listening... say 'Hey Nova' (Ctrl+C to stop)")

    try:
        while True:
            audio = sd.rec(config.WINDOW_SAMPLES, samplerate=config.SAMPLE_RATE,
                            channels=1, dtype="int16")
            sd.wait()
            waveform = audio.reshape(-1).astype(np.float32) / 32768.0
            waveform = normalize_amplitude(waveform)
            waveform = pad_or_crop(waveform)

            feats = waveform_to_mfcc_numpy(waveform)
            probs = predict(feats)

            line = "  ".join(f"{label}={p:.2f}" for label, p in zip(config.LABELS, probs))
            wake_p = probs[config.LABELS.index(config.WAKE_WORD)]
            marker = "  <-- WAKE!" if wake_p >= config.WAKE_THRESHOLD else ""
            log.info(f"{line}{marker}")
    except KeyboardInterrupt:
        log.info("stopped")


if __name__ == "__main__":
    main()
