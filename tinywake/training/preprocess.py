"""
training/preprocess.py

Loads raw WAVs from dataset/{hey_nova,unknown,noise,hard_negatives}/,
normalizes them to a common format, and produces a speaker-disjoint
train/val/test manifest (JSON) that train.py consumes.

Filenames are expected as: <speaker>_<index>.wav  (see collect_audio.py)
The speaker ID is parsed from the filename prefix.

IMPORTANT: splitting is done BY SPEAKER, not by individual file, to
avoid speaker leakage between train/val/test.
"""
from __future__ import annotations
import json
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("PREPROCESS")

# hard_negatives are phonetically-similar-but-wrong phrases; they are
# treated as class "unknown" for training, but tracked separately so
# evaluate.py can report false-accept rate on them specifically.
LABEL_DIRS = {
    "hey_nova": (config.DATASET_DIR / "hey_nova", "hey_nova"),
    "unknown": (config.DATASET_DIR / "unknown", "unknown"),
    "hard_negatives": (config.DATASET_DIR / "hard_negatives", "unknown"),
    "noise": (config.DATASET_DIR / "noise", "silence"),
}


def _speaker_from_filename(path: Path) -> str:
    # "<speaker>_<index>.wav" -> speaker (handles speaker names with underscores
    # by dropping only the trailing numeric token)
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return stem


def load_wav_mono16k(path: Path) -> np.ndarray:
    """Load a WAV file, resample to 16kHz mono float32 in [-1, 1] if needed."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAVs are supported (got {sample_width * 8}-bit)")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if sample_rate != config.SAMPLE_RATE:
        audio = _resample_linear(audio, sample_rate, config.SAMPLE_RATE)

    return audio


def _resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear-interpolation resampler (no SciPy dependency)."""
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    orig_x = np.linspace(0, duration, num=len(audio), endpoint=False)
    target_x = np.linspace(0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, orig_x, audio).astype(np.float32)


def normalize_amplitude(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.max(np.abs(audio)) + 1e-8
    return (audio * (target_peak / peak)).astype(np.float32)


def pad_or_crop(audio: np.ndarray, target_len: int = config.WINDOW_SAMPLES) -> np.ndarray:
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        # Center-crop
        start = (len(audio) - target_len) // 2
        return audio[start:start + target_len]
    pad_total = target_len - len(audio)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio, (pad_left, pad_right), mode="constant")


def build_manifest() -> Dict[str, List[dict]]:
    """
    Scans dataset/ and returns {"train": [...], "val": [...], "test": [...]}
    where each entry is {"path": str, "label": str, "speaker": str}.
    Split is performed per-speaker within each raw label directory.
    """
    speakers_by_label: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))

    for _, (dir_path, class_label) in LABEL_DIRS.items():
        if not dir_path.exists():
            continue
        for wav_path in sorted(dir_path.glob("*.wav")):
            speaker = _speaker_from_filename(wav_path)
            speakers_by_label[class_label][speaker].append(wav_path)

    manifest: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    rng = np.random.RandomState(42)

    for class_label, speakers in speakers_by_label.items():
        speaker_ids = sorted(speakers.keys())
        rng.shuffle(speaker_ids)

        n = len(speaker_ids)
        n_train = max(1, int(round(n * config.TRAIN_SPEAKER_SPLIT[0])))
        n_val = max(1, int(round(n * config.TRAIN_SPEAKER_SPLIT[1]))) if n > 2 else 0
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)

        train_speakers = set(speaker_ids[:n_train])
        val_speakers = set(speaker_ids[n_train:n_train + n_val])
        test_speakers = set(speaker_ids[n_train + n_val:])

        if n <= 2:
            # Too few speakers to split meaningfully — put everything in train
            # and warn; the caller should collect more speaker diversity.
            log.info(f"WARNING: class '{class_label}' has only {n} speaker(s); "
                      f"all data goes to train (add more speakers for val/test).")
            train_speakers = set(speaker_ids)
            val_speakers = set()
            test_speakers = set()

        for speaker, paths in speakers.items():
            if speaker in train_speakers:
                split = "train"
            elif speaker in val_speakers:
                split = "val"
            else:
                split = "test"
            for p in paths:
                manifest[split].append({
                    "path": str(p),
                    "label": class_label,
                    "speaker": speaker,
                })

    return manifest


def preprocess_waveform(path: str) -> np.ndarray:
    audio = load_wav_mono16k(Path(path))
    audio = normalize_amplitude(audio)
    audio = pad_or_crop(audio)
    return audio


def main() -> None:
    manifest = build_manifest()
    out_path = config.DATASET_DIR / "manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    for split, entries in manifest.items():
        speakers = {e["speaker"] for e in entries}
        labels = defaultdict(int)
        for e in entries:
            labels[e["label"]] += 1
        log.info(f"{split}: {len(entries)} samples, {len(speakers)} speakers, by-label={dict(labels)}")

    log.info(f"manifest written to {out_path}")


if __name__ == "__main__":
    main()
