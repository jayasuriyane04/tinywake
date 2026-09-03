"""
training/augmentation.py

Randomized waveform augmentation applied on-the-fly during training
(never materialized to disk as extra WAV files — see train.py's
tf.data pipeline, which calls `augment_waveform` inside a `.map()`).

All ops operate on float32 waveforms in roughly [-1, 1] at
config.SAMPLE_RATE, and are implemented with plain NumPy so they're
cheap enough to run per-batch on CPU.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _load_noise_bank() -> list[np.ndarray]:
    """Lazily load background noise clips from dataset/noise/ for additive
    noise augmentation. Falls back to synthetic white noise if none exist yet."""
    from training.preprocess import load_wav_mono16k  # local import, avoids cycle

    noise_dir = config.DATASET_DIR / "noise"
    clips = []
    if noise_dir.exists():
        for wav_path in sorted(noise_dir.glob("*.wav"))[:200]:
            try:
                clips.append(load_wav_mono16k(wav_path))
            except Exception:
                continue
    return clips


_NOISE_BANK: list[np.ndarray] | None = None


def _get_noise_bank() -> list[np.ndarray]:
    global _NOISE_BANK
    if _NOISE_BANK is None:
        _NOISE_BANK = _load_noise_bank()
    return _NOISE_BANK


def random_gain(audio: np.ndarray, rng: np.random.RandomState,
                 db_range=(-6.0, 6.0)) -> np.ndarray:
    gain_db = rng.uniform(*db_range)
    gain = 10 ** (gain_db / 20)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def random_time_shift(audio: np.ndarray, rng: np.random.RandomState,
                       max_shift_ms: float = 100.0) -> np.ndarray:
    max_shift = int(config.SAMPLE_RATE * max_shift_ms / 1000)
    shift = rng.randint(-max_shift, max_shift + 1)
    return np.roll(audio, shift).astype(np.float32)


def random_time_stretch(audio: np.ndarray, rng: np.random.RandomState,
                         rate_range=(0.9, 1.1)) -> np.ndarray:
    """Simple resample-based stretch (changes pitch slightly too — that's fine,
    it adds diversity), then pad/crop back to the original length."""
    rate = rng.uniform(*rate_range)
    orig_len = len(audio)
    new_len = max(1, int(orig_len / rate))
    x_old = np.linspace(0, 1, num=orig_len, endpoint=False)
    x_new = np.linspace(0, 1, num=new_len, endpoint=False)
    stretched = np.interp(x_new, x_old, audio).astype(np.float32)

    if len(stretched) >= orig_len:
        start = (len(stretched) - orig_len) // 2
        return stretched[start:start + orig_len]
    pad = orig_len - len(stretched)
    left = pad // 2
    return np.pad(stretched, (left, pad - left))


def random_pitch_shift(audio: np.ndarray, rng: np.random.RandomState,
                        semitone_range=(-2.0, 2.0)) -> np.ndarray:
    """Lightweight pitch shift via resampling + length correction
    (good enough for augmentation; not audio-quality-critical)."""
    semitones = rng.uniform(*semitone_range)
    rate = 2 ** (semitones / 12)
    orig_len = len(audio)
    resampled_len = max(1, int(orig_len / rate))
    x_old = np.linspace(0, 1, num=orig_len, endpoint=False)
    x_new = np.linspace(0, 1, num=resampled_len, endpoint=False)
    shifted = np.interp(x_new, x_old, audio).astype(np.float32)
    # resample back to original length to restore original duration/tempo
    x_old2 = np.linspace(0, 1, num=resampled_len, endpoint=False)
    x_new2 = np.linspace(0, 1, num=orig_len, endpoint=False)
    return np.interp(x_new2, x_old2, shifted).astype(np.float32)


def add_background_noise(audio: np.ndarray, rng: np.random.RandomState,
                          snr_db_range=(5.0, 20.0)) -> np.ndarray:
    bank = _get_noise_bank()
    if not bank:
        noise = rng.normal(0, 0.02, size=len(audio)).astype(np.float32)
    else:
        clip = bank[rng.randint(0, len(bank))]
        if len(clip) < len(audio):
            reps = int(np.ceil(len(audio) / len(clip)))
            clip = np.tile(clip, reps)
        start = rng.randint(0, max(1, len(clip) - len(audio) + 1))
        noise = clip[start:start + len(audio)]

    signal_power = np.mean(audio ** 2) + 1e-8
    noise_power = np.mean(noise ** 2) + 1e-8
    snr_db = rng.uniform(*snr_db_range)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)
    return np.clip(audio + noise * scale, -1.0, 1.0).astype(np.float32)


def add_simple_reverb(audio: np.ndarray, rng: np.random.RandomState,
                       decay_range=(0.15, 0.4), delay_ms_range=(20, 60)) -> np.ndarray:
    """Cheap single-tap reverb: audio + delayed decayed copy."""
    decay = rng.uniform(*decay_range)
    delay_samples = int(config.SAMPLE_RATE * rng.uniform(*delay_ms_range) / 1000)
    reverb = np.zeros_like(audio)
    reverb[delay_samples:] = audio[:-delay_samples] * decay
    return np.clip(audio + reverb, -1.0, 1.0).astype(np.float32)


def random_silence_insertion(audio: np.ndarray, rng: np.random.RandomState,
                              max_silence_ms: float = 150.0) -> np.ndarray:
    """Insert a short silent gap and crop back to original length, simulating
    hesitant/slow speech onset."""
    max_silence = int(config.SAMPLE_RATE * max_silence_ms / 1000)
    if max_silence <= 0:
        return audio
    silence_len = rng.randint(0, max_silence)
    insert_at = rng.randint(0, len(audio))
    silence = np.zeros(silence_len, dtype=np.float32)
    extended = np.concatenate([audio[:insert_at], silence, audio[insert_at:]])
    return pad_or_crop_to(extended, len(audio))


def pad_or_crop_to(audio: np.ndarray, target_len: int) -> np.ndarray:
    if len(audio) == target_len:
        return audio
    if len(audio) > target_len:
        start = (len(audio) - target_len) // 2
        return audio[start:start + target_len]
    pad = target_len - len(audio)
    left = pad // 2
    return np.pad(audio, (left, pad - left))


def augment_waveform(audio: np.ndarray, rng: np.random.RandomState | None = None,
                      p: float = 0.8) -> np.ndarray:
    """Apply a random subset of augmentations to one waveform. `p` is the
    probability that *any* augmentation is applied at all (keeps some clean
    examples in the mix)."""
    if rng is None:
        rng = np.random.RandomState()
    if rng.uniform() > p:
        return audio.astype(np.float32)

    out = audio.copy()
    if rng.uniform() < 0.6:
        out = random_gain(out, rng)
    if rng.uniform() < 0.5:
        out = random_time_shift(out, rng)
    if rng.uniform() < 0.3:
        out = random_time_stretch(out, rng)
    if rng.uniform() < 0.3:
        out = random_pitch_shift(out, rng)
    if rng.uniform() < 0.5:
        out = add_background_noise(out, rng)
    if rng.uniform() < 0.2:
        out = add_simple_reverb(out, rng)
    if rng.uniform() < 0.15:
        out = random_silence_insertion(out, rng)

    return pad_or_crop_to(out, len(audio)).astype(np.float32)
