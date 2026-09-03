"""
edge_simulator/main.py

Full laptop edge-device pipeline:

    microphone -> 20ms frames -> ring buffer (pre-roll)
               -> energy gate -> KWS inference every INFERENCE_INTERVAL_MS
               -> temporal smoothing / wake state machine
               -> on wake: send pre-roll + persistent WebSocket stream
               -> VAD-driven stream stop after silence
               -> back to LISTENING

Privacy guarantee: audio frames are only ever handed to
StreamingClient.send_audio() while kws.state == State.STREAMING.
While LISTENING (or in COOLDOWN), frames are consumed locally for
KWS inference only and never leave the process. See section 26 in
the project brief / README "Privacy" section.

Usage:
    python server/server.py            # in one terminal
    python edge_simulator/main.py      # in another
"""
from __future__ import annotations
import asyncio
import sys
import time
import uuid
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from edge_simulator.microphone import Microphone  # noqa: E402
from edge_simulator.ring_buffer import RingBuffer  # noqa: E402
from edge_simulator.kws_engine import KeywordSpotter, State  # noqa: E402
from edge_simulator.vad import VoiceActivityDetector, rms  # noqa: E402
from edge_simulator.streaming import StreamingClient  # noqa: E402

log = get_logger("MAIN")


class LatencyTracker:
    """Records T1-T6 timestamps for one wake-to-transcription cycle
    (see benchmarks/latency_test.py for offline analysis of saved CSVs)."""

    def __init__(self):
        self.t1_keyword_detected: float | None = None
        self.t2_streaming_started: float | None = None
        self.t3_first_packet_sent: float | None = None

    def reset(self) -> None:
        self.__init__()


async def run_pipeline(model_path: Path) -> None:
    mic = Microphone()
    ring = RingBuffer()
    kws = KeywordSpotter(model_path)
    vad = VoiceActivityDetector()
    client = StreamingClient()
    latency = LatencyTracker()

    await client.connect()

    mic.start()
    log.info("pipeline running — LISTENING (no audio leaves the device yet)")

    window_buffer = np.zeros(config.WINDOW_SAMPLES, dtype=np.int16)
    samples_since_inference = 0
    inference_interval_samples = int(config.SAMPLE_RATE * config.INFERENCE_INTERVAL_MS / 1000)

    def on_wake(score: float, t_detected: float) -> None:
        latency.reset()
        latency.t1_keyword_detected = t_detected

    kws.on_wake_detected = on_wake

    try:
        while True:
            frame = mic.read_frame(timeout=1.0)
            if frame is None:
                continue

            # Ring buffer always gets every frame — this is what lets us
            # recover pre-roll audio the instant a wake is confirmed.
            ring.push(frame)

            if kws.state in (State.LISTENING, State.POSSIBLE_WAKE, State.COOLDOWN):
                # Energy gate: skip inference on near-silent audio to save CPU
                # (this is the mechanism that keeps idle CPU low on ESP32).
                window_buffer = np.roll(window_buffer, -len(frame))
                window_buffer[-len(frame):] = frame
                samples_since_inference += len(frame)

                if samples_since_inference >= inference_interval_samples:
                    samples_since_inference = 0
                    frame_energy = rms(window_buffer[-len(frame):])
                    if frame_energy >= config.RMS_GATE_THRESHOLD:
                        kws.process_window(window_buffer.astype(np.float32) / 32768.0)
                    # else: below RMS_GATE_THRESHOLD -> skip inference entirely

            if kws.state == State.STREAMING:
                await _handle_streaming(kws, vad, client, ring, mic, latency)

    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        mic.stop()
        await client.close()


async def _handle_streaming(kws: KeywordSpotter, vad: VoiceActivityDetector,
                             client: StreamingClient, ring: RingBuffer,
                             mic: Microphone, latency: LatencyTracker) -> None:
    """Runs the full life of one streaming session: send pre-roll, stream
    live audio while VAD sees speech-or-recent-speech, then END_STREAM."""
    session_id = str(uuid.uuid4())[:8]
    log.info(f"[STREAM] session={session_id} started")

    latency.t2_streaming_started = time.monotonic()
    await client.send_start_stream(session_id)

    preroll = ring.read_all_chronological()
    log.info(f"[STREAM] preroll={config.PREROLL_MS}ms ({len(preroll)} samples)")
    await client.send_audio(preroll)
    latency.t3_first_packet_sent = time.monotonic()
    log.info("first packet sent")

    if latency.t1_keyword_detected is not None:
        edge_latency_ms = (latency.t2_streaming_started - latency.t1_keyword_detected) * 1000
        log.info(f"[LATENCY] edge_trigger_latency={edge_latency_ms:.1f}ms")

    vad.reset()
    vad.process_frame(preroll)  # count pre-roll toward speech detection too

    while True:
        frame = mic.read_frame(timeout=1.0)
        if frame is None:
            continue
        await client.send_audio(frame)
        vad.process_frame(frame)

        if vad.should_stop_streaming():
            log.info("[STREAM] silence detected")
            break

    await client.send_end_stream()
    kws.begin_cooldown()
    log.info("RETURNING TO LISTENING")


def main() -> None:
    model_path = config.MODELS_DIR / "kws_int8.tflite"
    if not model_path.exists():
        log.info(f"WARNING: {model_path} not found — falling back to kws_fp32.keras "
                  f"is not supported by kws_engine.py (TFLite only). "
                  f"Run training/quantize.py first.")
        sys.exit(1)

    asyncio.run(run_pipeline(model_path))


if __name__ == "__main__":
    main()
