"""
training/collect_audio.py

Records ~1s WAV samples from the microphone into dataset/<label>/ for
building the KWS training set.

Usage:
    python training/collect_audio.py --label hey_nova --speaker jay --samples 40
    python training/collect_audio.py --label unknown --speaker jay --samples 40
    python training/collect_audio.py --label hard_negative --speaker jay --samples 20
    python training/collect_audio.py --label noise --speaker room1 --samples 20
"""
from __future__ import annotations
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("COLLECT")

LABEL_TO_DIR = {
    "hey_nova": config.DATASET_DIR / "hey_nova",
    "unknown": config.DATASET_DIR / "unknown",
    "hard_negative": config.DATASET_DIR / "hard_negatives",
    "noise": config.DATASET_DIR / "noise",
}


def _next_index(out_dir: Path, speaker: str) -> int:
    existing = list(out_dir.glob(f"{speaker}_*.wav"))
    if not existing:
        return 0
    nums = []
    for p in existing:
        try:
            nums.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    return max(nums, default=-1) + 1


def _record_one(duration_s: float) -> np.ndarray:
    """Record `duration_s` seconds of mono int16 PCM at config.SAMPLE_RATE."""
    import sounddevice as sd

    frames = int(config.SAMPLE_RATE * duration_s)
    audio = sd.rec(frames, samplerate=config.SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)


def _save_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(config.CHANNELS)
        wf.setsampwidth(config.SAMPLE_WIDTH_BYTES)
        wf.setframerate(config.SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def _countdown(seconds: int = 2) -> None:
    for i in range(seconds, 0, -1):
        print(f"  recording in {i}...", end="\r")
        time.sleep(1)
    print("  GO! speak now         ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled wake-word dataset samples.")
    parser.add_argument("--label", required=True, choices=list(LABEL_TO_DIR.keys()))
    parser.add_argument("--speaker", required=True, help="Speaker ID, used in the filename.")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds per recording.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds between recordings.")
    args = parser.parse_args()

    out_dir = LABEL_TO_DIR[args.label]
    out_dir.mkdir(parents=True, exist_ok=True)

    start_idx = _next_index(out_dir, args.speaker)
    log.info(f"label={args.label} speaker={args.speaker} samples={args.samples} "
              f"starting_index={start_idx}")

    for i in range(args.samples):
        idx = start_idx + i
        print(f"\n[{i + 1}/{args.samples}] label='{args.label}' speaker='{args.speaker}'")
        _countdown(2)
        pcm = _record_one(args.duration)
        out_path = out_dir / f"{args.speaker}_{idx:04d}.wav"
        _save_wav(out_path, pcm)
        log.info(f"saved {out_path.relative_to(config.ROOT_DIR)}")
        time.sleep(args.pause)

    log.info(f"done — {args.samples} recordings saved to {out_dir}")


if __name__ == "__main__":
    main()
