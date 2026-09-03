"""
server/session.py

Per-connection session state: tracks the current stream (if any),
accumulated audio, and the T4/T5/T6 latency timestamps for the
active utterance.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Session:
    session_id: str
    client_id: str
    is_streaming: bool = False
    audio_chunks: list = field(default_factory=list)
    t4_first_packet_received: float | None = None
    t5_asr_started: float | None = None
    t6_transcription_completed: float | None = None
    packets_received: int = 0
    bytes_received: int = 0

    def start_stream(self, session_id: str) -> None:
        self.session_id = session_id
        self.is_streaming = True
        self.audio_chunks = []
        self.t4_first_packet_received = None
        self.t5_asr_started = None
        self.t6_transcription_completed = None
        self.packets_received = 0
        self.bytes_received = 0

    def add_audio(self, pcm_bytes: bytes) -> None:
        if self.t4_first_packet_received is None:
            self.t4_first_packet_received = time.monotonic()
        self.audio_chunks.append(np.frombuffer(pcm_bytes, dtype="<i2"))
        self.packets_received += 1
        self.bytes_received += len(pcm_bytes)

    def end_stream(self) -> np.ndarray:
        self.is_streaming = False
        if self.audio_chunks:
            full_audio = np.concatenate(self.audio_chunks)
        else:
            full_audio = np.array([], dtype=np.int16)
        return full_audio
