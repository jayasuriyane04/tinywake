"""
tinywake/config.py

Single source of truth for every tunable constant in the project.
No module should hardcode sample rates, thresholds, or timing values —
import them from here so training, the edge simulator, and the server
all agree on the same numbers (and so the numbers are easy to re-tune
for the eventual ESP32-S3 port).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
BENCHMARK_RESULTS_DIR = ROOT_DIR / "benchmarks" / "results"

for _d in (MODELS_DIR, LOGS_DIR, BENCHMARK_RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

WAKE_WORD = "hey_nova"
LABELS = ["silence", "unknown", "hey_nova"]  # index 0, 1, 2 — must match model.py

# --------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------
SAMPLE_RATE = 16000          # Hz, mono, signed 16-bit PCM everywhere
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2       # int16
WINDOW_SECONDS = 1.0         # duration of one classification window
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)
FRAME_MS = 20                # size of one microphone read / ring-buffer push
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# --------------------------------------------------------------------------
# Feature extraction (MFCC)
# --------------------------------------------------------------------------
MFCC_WINDOW_MS = 30
MFCC_STRIDE_MS = 20
MFCC_COUNT = 13
# ~ (1000 - 30) / 20 + 1 = 49 frames for a 1s window
MFCC_NUM_FRAMES = int((WINDOW_SECONDS * 1000 - MFCC_WINDOW_MS) / MFCC_STRIDE_MS) + 1
FEATURE_SHAPE = (MFCC_NUM_FRAMES, MFCC_COUNT, 1)  # (49, 13, 1)

LOWER_HERTZ = 80.0
UPPER_HERTZ = 7600.0
NUM_MEL_BINS = 40

# --------------------------------------------------------------------------
# KWS decision logic
# --------------------------------------------------------------------------
WAKE_THRESHOLD = 0.90
CONSECUTIVE_HITS = 3          # min consecutive frames above threshold
SMOOTHING_WINDOW = 3           # rolling average window for scores
COOLDOWN_MS = 1500             # ignore new wake triggers for this long after one fires
INFERENCE_INTERVAL_MS = 100    # run KWS inference this often (not every 20ms frame)

# --------------------------------------------------------------------------
# Energy gate (cheap pre-filter before running the model)
# --------------------------------------------------------------------------
RMS_GATE_THRESHOLD = 150.0     # int16 RMS; below this, skip inference entirely

# --------------------------------------------------------------------------
# Ring buffer / pre-roll
# --------------------------------------------------------------------------
PREROLL_MS = 300

# --------------------------------------------------------------------------
# VAD / streaming
# --------------------------------------------------------------------------
VAD_RMS_THRESHOLD = 200.0
SILENCE_TIMEOUT_MS = 800

# --------------------------------------------------------------------------
# Networking
# --------------------------------------------------------------------------
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8000
WEBSOCKET_URL = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}/ws"
DASHBOARD_WS_URL = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}/dashboard"

# --------------------------------------------------------------------------
# ASR
# --------------------------------------------------------------------------
ASR_ENGINE = "faster-whisper"
ASR_MODEL_SIZE = "tiny.en"     # small for low demo latency; swap to "base" etc.
ASR_DEVICE = "cpu"
ASR_COMPUTE_TYPE = "int8"

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
TRAIN_SPEAKER_SPLIT = (0.70, 0.15, 0.15)  # train / val / test, split BY SPEAKER
BATCH_SIZE = 32
EPOCHS = 60
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 8

CANDIDATE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]


@dataclass
class PacketFlags:
    """Bit flags for the streaming.py binary packet header (see streaming.py)."""
    START_STREAM: int = 0x01
    AUDIO: int = 0x02
    END_STREAM: int = 0x04


FLAGS = PacketFlags()
