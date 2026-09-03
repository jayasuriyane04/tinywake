"""
server/server.py

FastAPI + WebSocket ASR server.

Two endpoints:
    ws://.../ws          — edge device clients (binary audio packets)
    ws://.../dashboard    — dashboard clients (receive JSON status/event broadcasts)

Flow per edge connection:
    1. Client connects, socket stays open indefinitely (persistent connection).
    2. START_STREAM packet -> new Session, T4 timer starts on first AUDIO packet.
    3. AUDIO packets -> buffered into the session.
    4. END_STREAM packet -> full utterance handed to ASR, result sent back to
       the edge client AND broadcast to all connected dashboards.

Run:
    python server/server.py
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from server.session import Session  # noqa: E402
from server.audio_receiver import handle_packet  # noqa: E402
from server.asr import build_asr_engine  # noqa: E402

log = get_logger("SERVER")

app = FastAPI(title="tinywake ASR server")

_asr_engine = None
_dashboard_clients: set[WebSocket] = set()
_stats = {
    "total_wake_events": 0,
    "total_transcriptions": 0,
}


def get_asr_engine():
    global _asr_engine
    if _asr_engine is None:
        _asr_engine = build_asr_engine()
    return _asr_engine


async def _broadcast_to_dashboards(message: dict) -> None:
    if not _dashboard_clients:
        return
    payload = json.dumps(message)
    dead = []
    for ws in _dashboard_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _dashboard_clients.discard(ws)


@app.websocket("/ws")
async def edge_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    client_id = str(uuid.uuid4())[:8]
    session = Session(session_id="", client_id=client_id)
    log.info(f"edge client connected: {client_id}")

    await _broadcast_to_dashboards({"type": "client_connected", "client_id": client_id})

    try:
        while True:
            raw = await websocket.receive_bytes()
            kind = handle_packet(session, raw)

            if kind == "start":
                _stats["total_wake_events"] += 1
                await _broadcast_to_dashboards({
                    "type": "wake_detected",
                    "session_id": session.session_id,
                    "client_id": client_id,
                })

            elif kind == "end":
                audio = session.end_stream()
                await _process_utterance(websocket, session, audio, client_id)

    except WebSocketDisconnect:
        log.info(f"edge client disconnected: {client_id}")
        await _broadcast_to_dashboards({"type": "client_disconnected", "client_id": client_id})


async def _process_utterance(websocket: WebSocket, session: Session,
                              audio: np.ndarray, client_id: str) -> None:
    if len(audio) == 0:
        log.info("WARNING: empty utterance, skipping ASR")
        return

    session.t5_asr_started = time.monotonic()
    engine = get_asr_engine()

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, engine.transcribe, audio, config.SAMPLE_RATE)

    session.t6_transcription_completed = time.monotonic()
    _stats["total_transcriptions"] += 1

    text = result["text"]
    log.info(f"Transcription: {text}" if text else "Transcription: (empty)")

    latencies = _compute_latencies(session)

    response = {
        "type": "transcription",
        "session_id": session.session_id,
        "client_id": client_id,
        "text": text,
        "language": result["language"],
        "processing_ms": result["processing_ms"],
        "latencies_ms": latencies,
        "packets_received": session.packets_received,
        "bytes_received": session.bytes_received,
    }

    await websocket.send_text(json.dumps(response))
    await _broadcast_to_dashboards(response)


def _compute_latencies(session: Session) -> dict:
    latencies = {}
    if session.t4_first_packet_received and session.t5_asr_started:
        latencies["wake_to_server_ms"] = (session.t5_asr_started - session.t4_first_packet_received) * 1000
    if session.t5_asr_started and session.t6_transcription_completed:
        latencies["asr_latency_ms"] = (session.t6_transcription_completed - session.t5_asr_started) * 1000
    if session.t4_first_packet_received and session.t6_transcription_completed:
        latencies["total_server_latency_ms"] = (
            session.t6_transcription_completed - session.t4_first_packet_received
        ) * 1000
    return latencies


@app.websocket("/dashboard")
async def dashboard_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _dashboard_clients.add(websocket)
    log.info("dashboard client connected")
    try:
        await websocket.send_text(json.dumps({"type": "stats", **_stats}))
        while True:
            await websocket.receive_text()  # dashboards don't send meaningful data; keep alive
    except WebSocketDisconnect:
        _dashboard_clients.discard(websocket)
        log.info("dashboard client disconnected")


@app.get("/health")
async def health():
    return {"status": "ok", **_stats}


def main() -> None:
    log.info(f"starting server on {config.WEBSOCKET_HOST}:{config.WEBSOCKET_PORT}")
    uvicorn.run(app, host=config.WEBSOCKET_HOST, port=config.WEBSOCKET_PORT, log_level="warning")


if __name__ == "__main__":
    main()
