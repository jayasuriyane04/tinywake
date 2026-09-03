"""
edge_simulator/microphone.py

Continuous microphone capture, delivered as fixed-size int16 PCM
frames (config.FRAME_MS each). This is the one module a future
ESP32-S3 port replaces wholesale with I2S DMA reads — see
docs/esp32_mapping.md.

Design constraint: never buffer more than a couple of frames ahead
of the consumer. On embedded hardware this maps to a small I2S DMA
ring with 2-4 buffers; we mirror that here with a bounded queue
instead of letting sounddevice's callback pile up unbounded audio.
"""
from __future__ import annotations
import queue
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("MIC")


class Microphone:
    """Pull-based microphone reader yielding FRAME_SAMPLES int16 frames."""

    def __init__(self, frame_samples: int = config.FRAME_SAMPLES,
                 sample_rate: int = config.SAMPLE_RATE, max_queue_frames: int = 8):
        self.frame_samples = frame_samples
        self.sample_rate = sample_rate
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=max_queue_frames)
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.info(f"stream status: {status}")
        frame = indata[:, 0].copy()
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame rather than blocking the audio callback —
            # mirrors what a small embedded DMA ring does under overrun.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(frame)

    def start(self) -> None:
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.frame_samples,
            callback=self._callback,
        )
        self._stream.start()
        log.info(f"microphone started: {self.sample_rate}Hz mono, "
                  f"{self.frame_samples} samples/frame ({config.FRAME_MS}ms)")

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            log.info("microphone stopped")

    def read_frame(self, timeout: Optional[float] = 1.0) -> Optional[np.ndarray]:
        """Blocking read of the next FRAME_SAMPLES int16 frame, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def frames(self):
        """Generator form, for `for frame in mic.frames(): ...` usage."""
        while True:
            frame = self.read_frame()
            if frame is not None:
                yield frame

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
