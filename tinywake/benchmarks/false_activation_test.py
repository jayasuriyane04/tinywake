"""
benchmarks/false_activation_test.py

Feeds long negative audio recordings (noise / hard_negatives / any
WAV with no wake word) through the KWS engine to measure False
Activations Per Hour (FAPH) — one of the project's primary metrics.

Usage:
    python benchmarks/false_activation_test.py --dir dataset/noise
    python benchmarks/false_activation_test.py --dir dataset/hard_negatives
    python benchmarks/false_activation_test.py --file /path/to/long_recording.wav
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from training.preprocess import load_wav_mono16k  # noqa: E402
from edge_simulator.kws_engine import KeywordSpotter, State  # noqa: E402

log = get_logger("FAPH")


def _iter_frames(audio: np.ndarray, frame_samples: int = config.FRAME_SAMPLES):
    for start in range(0, len(audio) - frame_samples + 1, frame_samples):
        yield audio[start:start + frame_samples]


def run_faph_test(wav_paths: list[Path], model_path: Path) -> dict:
    kws = KeywordSpotter(model_path)

    activations = [0]

    def on_wake(score: float, t_detected: float) -> None:
        activations[0] += 1
        log.info(f"FALSE ACTIVATION #{activations[0]} (score={score:.2f})")

    kws.on_wake_detected = on_wake

    total_seconds = 0.0
    window_buffer = np.zeros(config.WINDOW_SAMPLES, dtype=np.float32)
    inference_interval_samples = int(config.SAMPLE_RATE * config.INFERENCE_INTERVAL_MS / 1000)

    for wav_path in wav_paths:
        audio_f32 = load_wav_mono16k(wav_path)
        total_seconds += len(audio_f32) / config.SAMPLE_RATE
        samples_since_inference = 0

        for i in range(0, len(audio_f32), config.FRAME_SAMPLES):
            frame = audio_f32[i:i + config.FRAME_SAMPLES]
            if len(frame) == 0:
                continue
            window_buffer = np.roll(window_buffer, -len(frame))
            window_buffer[-len(frame):] = frame
            samples_since_inference += len(frame)

            if samples_since_inference >= inference_interval_samples:
                samples_since_inference = 0
                kws.process_window(window_buffer)
                if kws.state == State.STREAMING:
                    kws.begin_cooldown()  # simulate returning to listening immediately

    hours = total_seconds / 3600
    faph = activations[0] / hours if hours > 0 else float("nan")

    log.info(f"total audio: {total_seconds:.1f}s ({hours:.3f} hours)")
    log.info(f"false activations: {activations[0]}")
    log.info(f"False Activations Per Hour (FAPH): {faph:.2f}")

    return {
        "total_seconds": total_seconds,
        "false_activations": activations[0],
        "faph": faph,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None, help="Directory of negative WAVs.")
    parser.add_argument("--file", type=str, default=None, help="Single long negative WAV.")
    parser.add_argument("--model", type=str, default=str(config.MODELS_DIR / "kws_int8.tflite"))
    args = parser.parse_args()

    if args.dir:
        wav_paths = sorted(Path(args.dir).glob("*.wav"))
    elif args.file:
        wav_paths = [Path(args.file)]
    else:
        # Default: run against everything in noise/ and hard_negatives/
        wav_paths = sorted((config.DATASET_DIR / "noise").glob("*.wav")) + \
                    sorted((config.DATASET_DIR / "hard_negatives").glob("*.wav"))

    if not wav_paths:
        log.info("no negative audio files found — collect some with collect_audio.py "
                  "(--label noise / --label hard_negative)")
        sys.exit(1)

    run_faph_test(wav_paths, Path(args.model))


if __name__ == "__main__":
    main()
