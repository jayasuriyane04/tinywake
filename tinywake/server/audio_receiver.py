"""
server/audio_receiver.py

Server-side counterpart to edge_simulator/streaming.py's packet
framing. Parses incoming binary WebSocket frames into
(seq, timestamp_ms, flags, payload) and routes them into a Session.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from edge_simulator.streaming import unpack_packet  # noqa: E402
from server.session import Session  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("SERVER")


def handle_packet(session: Session, raw: bytes) -> str:
    """Parses one raw binary WS message and mutates `session` accordingly.
    Returns one of: "start", "audio", "end" for the caller to react to."""
    seq, timestamp_ms, payload_size, flags, payload = unpack_packet(raw)

    if flags & config.FLAGS.START_STREAM:
        session_id = payload.decode("utf-8") if payload else session.session_id
        session.start_stream(session_id)
        log.info(f"first packet received latency=0ms session={session_id}")
        return "start"

    if flags & config.FLAGS.AUDIO:
        first = session.t4_first_packet_received is None
        session.add_audio(payload)
        if first:
            log.info("first packet received")
        return "audio"

    if flags & config.FLAGS.END_STREAM:
        return "end"

    log.info(f"WARNING: unknown packet flags={flags} seq={seq}")
    return "unknown"
