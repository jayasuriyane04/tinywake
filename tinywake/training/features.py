"""
training/features.py

Waveform -> MFCC feature extraction.

Implemented with tf.signal so that the exact same math can, in
principle, be re-implemented on-device (ESP32 embedded MFCC) with a
matching numeric pipeline. Keep this file free of anything that
wouldn't have a reasonable embedded equivalent (no librosa, no SciPy
filters) — it's the one module in `training/` that doubles as a
spec for `features.py -> embedded MFCC` (see docs/esp32_mapping.md).

Output shape for a 1-second, 16kHz window: (49, 13, 1) — see
config.FEATURE_SHAPE.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def _stft_frame_params():
    frame_length = int(config.SAMPLE_RATE * config.MFCC_WINDOW_MS / 1000)
    frame_step = int(config.SAMPLE_RATE * config.MFCC_STRIDE_MS / 1000)
    fft_length = 1
    while fft_length < frame_length:
        fft_length *= 2
    return frame_length, frame_step, fft_length


_FRAME_LENGTH, _FRAME_STEP, _FFT_LENGTH = _stft_frame_params()

_MEL_WEIGHT_MATRIX = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=config.NUM_MEL_BINS,
    num_spectrogram_bins=_FFT_LENGTH // 2 + 1,
    sample_rate=config.SAMPLE_RATE,
    lower_edge_hertz=config.LOWER_HERTZ,
    upper_edge_hertz=config.UPPER_HERTZ,
)


def waveform_to_mfcc(waveform: tf.Tensor) -> tf.Tensor:
    """
    waveform: float32 tensor, shape (num_samples,), values in [-1, 1],
              expected length == config.WINDOW_SAMPLES (pad/crop upstream).

    returns: float32 tensor, shape config.FEATURE_SHAPE == (49, 13, 1)
    """
    waveform = tf.cast(waveform, tf.float32)

    stfts = tf.signal.stft(
        waveform,
        frame_length=_FRAME_LENGTH,
        frame_step=_FRAME_STEP,
        fft_length=_FFT_LENGTH,
        window_fn=tf.signal.hann_window,
    )
    spectrograms = tf.abs(stfts)

    mel_spectrograms = tf.tensordot(spectrograms, _MEL_WEIGHT_MATRIX, 1)
    mel_spectrograms.set_shape(
        spectrograms.shape[:-1].concatenate(_MEL_WEIGHT_MATRIX.shape[-1:])
    )
    log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
    mfccs = mfccs[..., : config.MFCC_COUNT]  # keep lowest N coefficients

    # Pad/crop the time axis to the exact expected number of frames —
    # STFT frame count can be off by one depending on input length.
    num_frames = tf.shape(mfccs)[0]
    target_frames = config.MFCC_NUM_FRAMES

    def pad():
        pad_amount = target_frames - num_frames
        return tf.pad(mfccs, [[0, pad_amount], [0, 0]])

    def crop():
        return mfccs[:target_frames, :]

    mfccs = tf.cond(num_frames < target_frames, pad, crop)
    mfccs.set_shape([target_frames, config.MFCC_COUNT])

    # Per-utterance mean/variance normalization — cheap and makes the
    # model far more robust to mic gain / distance differences.
    mean = tf.reduce_mean(mfccs)
    std = tf.math.reduce_std(mfccs) + 1e-6
    mfccs = (mfccs - mean) / std

    return tf.expand_dims(mfccs, axis=-1)  # (49, 13, 1)


def waveform_to_mfcc_numpy(waveform: np.ndarray) -> np.ndarray:
    """Convenience wrapper for callers outside the tf.data pipeline
    (e.g. the edge simulator running a single window at a time)."""
    return waveform_to_mfcc(tf.constant(waveform, dtype=tf.float32)).numpy()


if __name__ == "__main__":
    # Quick self-check.
    dummy = np.random.uniform(-0.1, 0.1, size=config.WINDOW_SAMPLES).astype(np.float32)
    feats = waveform_to_mfcc_numpy(dummy)
    print(f"Input samples: {dummy.shape}")
    print(f"Feature shape: {feats.shape} (expected {config.FEATURE_SHAPE})")
    assert feats.shape == config.FEATURE_SHAPE
    print("OK")
