"""
edge_simulator/ring_buffer.py

Fixed-capacity circular buffer of int16 PCM audio, sized to hold
config.PREROLL_MS of history. When the wake word fires, the caller
pulls the pre-roll out of here so the beginning of the user's
command (spoken right as/before detection confirms) isn't lost.

Implemented as a single pre-allocated NumPy array with a write
pointer — no dynamic allocation after construction, no Python lists
of chunks. This is deliberately the same shape as a static
C circular buffer so it maps directly onto ring_buffer.py -> static
circular buffer on the ESP32 (see docs/esp32_mapping.md).

Memory footprint: PREROLL_MS(300) * SAMPLE_RATE(16000) / 1000 * 2 bytes
                 = ~9.6 KB — trivial even on a 256KB-RAM target.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class RingBuffer:
    def __init__(self, duration_ms: int = config.PREROLL_MS,
                 sample_rate: int = config.SAMPLE_RATE):
        self.capacity = int(sample_rate * duration_ms / 1000)
        self._buf = np.zeros(self.capacity, dtype=np.int16)
        self._write_pos = 0
        self._filled = 0

    @property
    def size_bytes(self) -> int:
        return self._buf.nbytes

    def push(self, frame: np.ndarray) -> None:
        """Write an int16 frame into the ring, wrapping as needed.
        Frames longer than the capacity are truncated to the most recent tail."""
        n = len(frame)
        if n >= self.capacity:
            self._buf[:] = frame[-self.capacity:]
            self._write_pos = 0
            self._filled = self.capacity
            return

        end = self._write_pos + n
        if end <= self.capacity:
            self._buf[self._write_pos:end] = frame
        else:
            first_part = self.capacity - self._write_pos
            self._buf[self._write_pos:] = frame[:first_part]
            self._buf[:end - self.capacity] = frame[first_part:]

        self._write_pos = end % self.capacity
        self._filled = min(self.capacity, self._filled + n)

    def read_all_chronological(self) -> np.ndarray:
        """Return the buffered audio in correct time order (oldest -> newest)."""
        if self._filled < self.capacity:
            return self._buf[:self._filled].copy()
        return np.concatenate([self._buf[self._write_pos:], self._buf[:self._write_pos]])

    def clear(self) -> None:
        self._buf[:] = 0
        self._write_pos = 0
        self._filled = 0
