"""
edge_simulator/kws_engine.py

Continuous keyword-spotting engine. Runs INT8 TFLite inference on
rolling 1s windows every INFERENCE_INTERVAL_MS, smooths scores over
a short rolling window, and requires CONSECUTIVE_HITS above
WAKE_THRESHOLD before declaring a wake event — a single high-scoring
frame is never enough (avoids one-shot false triggers).

State machine (see kws_engine.State):

    LISTENING -> POSSIBLE_WAKE -> CONFIRMED_WAKE -> STREAMING -> COOLDOWN -> LISTENING

All transitions are logged via the [KWS]/[WAKE] tags.
"""
from __future__ import annotations
import collections
import enum
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from training.features import waveform_to_mfcc_numpy  # noqa: E402
from training.preprocess import normalize_amplitude, pad_or_crop  # noqa: E402

log = get_logger("KWS")
wake_log = get_logger("WAKE")

WAKE_INDEX = config.LABELS.index(config.WAKE_WORD)


class State(enum.Enum):
    LISTENING = "LISTENING"
    POSSIBLE_WAKE = "POSSIBLE_WAKE"
    CONFIRMED_WAKE = "CONFIRMED_WAKE"
    STREAMING = "STREAMING"
    COOLDOWN = "COOLDOWN"


class TFLiteKWSModel:
    """Thin wrapper around a TFLite Interpreter handling INT8 quantized I/O."""

    def __init__(self, model_path: Path):
        import tensorflow as tf
        self._interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._in_detail = self._interpreter.get_input_details()[0]
        self._out_detail = self._interpreter.get_output_details()[0]
        self._in_scale, self._in_zero = self._in_detail["quantization"]
        self._out_scale, self._out_zero = self._out_detail["quantization"]

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = features[np.newaxis, ...]
        if self._in_detail["dtype"] == np.int8:
            x = (x / self._in_scale + self._in_zero).round().astype(np.int8)
        else:
            x = x.astype(np.float32)
        self._interpreter.set_tensor(self._in_detail["index"], x)
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._out_detail["index"])[0]
        if self._out_detail["dtype"] == np.int8:
            out = (out.astype(np.float32) - self._out_zero) * self._out_scale
        return out


class KeywordSpotter:
    def __init__(self, model_path: Path,
                 wake_threshold: float = config.WAKE_THRESHOLD,
                 consecutive_hits: int = config.CONSECUTIVE_HITS,
                 smoothing_window: int = config.SMOOTHING_WINDOW,
                 cooldown_ms: int = config.COOLDOWN_MS):
        self.model = TFLiteKWSModel(model_path)
        self.wake_threshold = wake_threshold
        self.consecutive_hits = consecutive_hits
        self.cooldown_ms = cooldown_ms

        self._recent_scores: collections.deque[float] = collections.deque(maxlen=smoothing_window)
        self._consecutive_above_threshold = 0
        self.state = State.LISTENING
        self._cooldown_until: Optional[float] = None

        self.on_wake_detected = None  # optional callback: fn(score: float, t_detected: float)

    def _smoothed_score(self, raw_score: float) -> float:
        self._recent_scores.append(raw_score)
        return float(np.mean(self._recent_scores))

    def _set_state(self, new_state: State) -> None:
        if new_state != self.state:
            log.info(f"state {self.state.value} -> {new_state.value}")
            self.state = new_state

    def process_window(self, waveform: np.ndarray) -> float:
        """Run one classification pass over a WINDOW_SAMPLES waveform,
        update the internal state machine, and return the smoothed
        wake-word probability. Call this roughly every
        INFERENCE_INTERVAL_MS, not every single audio frame."""
        now = time.monotonic()

        if self.state == State.COOLDOWN:
            if self._cooldown_until is not None and now >= self._cooldown_until:
                self._set_state(State.LISTENING)
                self._consecutive_above_threshold = 0
            else:
                return 0.0  # ignore inference results during cooldown

        waveform = normalize_amplitude(waveform)
        waveform = pad_or_crop(waveform)
        features = waveform_to_mfcc_numpy(waveform)
        probs = self.model.predict(features)
        raw_score = float(probs[WAKE_INDEX])
        score = self._smoothed_score(raw_score)

        log.info(f"score={score:.2f}")

        if score >= self.wake_threshold:
            self._consecutive_above_threshold += 1
            if self.state == State.LISTENING and self._consecutive_above_threshold >= 1:
                self._set_state(State.POSSIBLE_WAKE)
            if self._consecutive_above_threshold >= self.consecutive_hits:
                self._confirm_wake(score, now)
        else:
            self._consecutive_above_threshold = 0
            if self.state == State.POSSIBLE_WAKE:
                self._set_state(State.LISTENING)

        return score

    def _confirm_wake(self, score: float, now: float) -> None:
        self._set_state(State.CONFIRMED_WAKE)
        wake_log.info(f"Hey Nova detected (confidence={score:.2f})")
        if self.on_wake_detected is not None:
            self.on_wake_detected(score, now)
        self._set_state(State.STREAMING)

    def begin_cooldown(self) -> None:
        self._set_state(State.COOLDOWN)
        self._cooldown_until = time.monotonic() + self.cooldown_ms / 1000
        self._consecutive_above_threshold = 0
        self._recent_scores.clear()
