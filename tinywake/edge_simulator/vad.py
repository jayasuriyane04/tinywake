"""
edge_simulator/vad.py

Lightweight energy-based Voice Activity Detector used to decide when
to stop streaming after the wake word (i.e. when the user has
finished their command).

Not a speech-model VAD (e.g. WebRTC VAD or Silero) on purpose — RMS
thresholding is the cheapest thing that works acceptably and is the
easiest to port to an embedded target with no ML model at all.

Usage pattern:
    vad = VoiceActivityDetector()
    vad.reset()
    for frame in mic.frames():
        is_speech = vad.process_frame(frame)
        if vad.should_stop_streaming():
            break
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("STREAM")


def rms(frame: np.ndarray) -> float:
    return float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)))


class VoiceActivityDetector:
    def __init__(self, rms_threshold: float = config.VAD_RMS_THRESHOLD,
                 silence_timeout_ms: int = config.SILENCE_TIMEOUT_MS):
        self.rms_threshold = rms_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self._last_speech_time: Optional[float] = None
        self._speech_started = False

    def reset(self) -> None:
        self._last_speech_time = time.monotonic()
        self._speech_started = False
        log.info(f"VAD reset (silence_timeout={self.silence_timeout_ms}ms)")

    def process_frame(self, frame: np.ndarray) -> bool:
        """Feed one audio frame; returns True if this frame is classified
        as speech (energy above threshold)."""
        energy = rms(frame)
        is_speech = energy >= self.rms_threshold
        now = time.monotonic()

        if is_speech:
            if not self._speech_started:
                self._speech_started = True
                log.info("speech")
            self._last_speech_time = now
        return is_speech

    def silence_elapsed_ms(self) -> float:
        if self._last_speech_time is None:
            return 0.0
        return (time.monotonic() - self._last_speech_time) * 1000

    def should_stop_streaming(self) -> bool:
        """True once we've seen at least one speech frame and then
        silence_timeout_ms of continuous silence."""
        if not self._speech_started:
            return False
        return self.silence_elapsed_ms() >= self.silence_timeout_ms
