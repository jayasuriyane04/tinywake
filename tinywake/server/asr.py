"""
server/asr.py

ASR engine abstraction. `FasterWhisperEngine` is the default
implementation; a future `WhisperCppEngine` can implement the same
`ASREngine` interface (e.g. shelling out to a whisper.cpp binary or
binding via pywhispercpp) without touching server.py.
"""
from __future__ import annotations
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("ASR")


class TranscriptionResult(TypedDict):
    text: str
    language: str
    processing_ms: float


class ASREngine(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> TranscriptionResult:
        """audio: int16 or float32 mono PCM array."""
        raise NotImplementedError


class FasterWhisperEngine(ASREngine):
    def __init__(self, model_size: str = config.ASR_MODEL_SIZE,
                 device: str = config.ASR_DEVICE,
                 compute_type: str = config.ASR_COMPUTE_TYPE):
        from faster_whisper import WhisperModel
        log.info(f"loading faster-whisper model='{model_size}' device={device} "
                  f"compute_type={compute_type} ...")
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)
        log.info("faster-whisper model loaded")

    def transcribe(self, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> TranscriptionResult:
        if audio.dtype == np.int16:
            audio_f32 = audio.astype(np.float32) / 32768.0
        else:
            audio_f32 = audio.astype(np.float32)

        start = time.monotonic()
        segments, info = self._model.transcribe(audio_f32, language=None, beam_size=1)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        elapsed_ms = (time.monotonic() - start) * 1000

        return {
            "text": text,
            "language": getattr(info, "language", "unknown"),
            "processing_ms": elapsed_ms,
        }


class MockASREngine(ASREngine):
    """No-dependency fallback used by tests / demos without faster-whisper
    installed. Returns a canned response instantly."""

    def transcribe(self, audio: np.ndarray, sample_rate: int = config.SAMPLE_RATE) -> TranscriptionResult:
        duration_s = len(audio) / sample_rate
        return {
            "text": f"[mock transcription — {duration_s:.1f}s of audio]",
            "language": "en",
            "processing_ms": 0.0,
        }


def build_asr_engine() -> ASREngine:
    try:
        return FasterWhisperEngine()
    except Exception as e:  # pragma: no cover - environment dependent
        log.info(f"WARNING: could not load faster-whisper ({e}); using MockASREngine")
        return MockASREngine()
