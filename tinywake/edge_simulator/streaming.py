"""
edge_simulator/streaming.py

Maintains a persistent WebSocket connection to the ASR server (opened
at startup, kept alive through LISTENING — no audio is sent while
listening, but the socket itself is not torn down and reopened per
wake event, since that adds needless connect latency right when we
want to be fast).

Binary packet format (little-endian), header + payload:

    offset  size  field
    0       4     sequence number (uint32)
    4       8     timestamp_ms (float64, time.time()*1000 at send)
    12      4     payload_size (uint32, bytes of PCM16 payload)
    16      1     flags (uint8) — see config.FLAGS / PacketFlags
    17      ...   payload (raw PCM16 bytes, empty for START/END control packets)

Kept intentionally simple (fixed struct header) so it can be
re-implemented in C for the ESP-IDF WebSocket client with no
behavioral changes — see docs/esp32_mapping.md.

Designed so PCM16 can later be swapped for ADPCM/Opus by changing
only how `payload` bytes are produced before calling send_audio().
"""
from __future__ import annotations
import struct
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("NETWORK")

_HEADER_FORMAT = "<IdIB"  # seq(u32), timestamp_ms(f64), payload_size(u32), flags(u8)
_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)


def pack_packet(seq: int, flags: int, payload: bytes = b"") -> bytes:
    header = struct.pack(_HEADER_FORMAT, seq, time.time() * 1000, len(payload), flags)
    return header + payload


def unpack_packet(data: bytes) -> tuple[int, float, int, int, bytes]:
    seq, timestamp_ms, payload_size, flags = struct.unpack(_HEADER_FORMAT, data[:_HEADER_SIZE])
    payload = data[_HEADER_SIZE:_HEADER_SIZE + payload_size]
    return seq, timestamp_ms, payload_size, flags, payload


class StreamingClient:
    """Synchronous-friendly wrapper around a `websockets` connection,
    designed to be driven from an asyncio event loop (see main.py)."""

    def __init__(self, url: str = config.WEBSOCKET_URL):
        self.url = url
        self._ws = None
        self._seq = 0
        self.session_id: Optional[str] = None
        self.bytes_sent = 0
        self.packets_sent = 0

    async def connect(self) -> None:
        import websockets
        self._ws = await websockets.connect(self.url, max_size=None)
        log.info(f"connected to {self.url}")

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            log.info("connection closed")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def send_start_stream(self, session_id: str) -> None:
        self.session_id = session_id
        packet = pack_packet(self._next_seq(), config.FLAGS.START_STREAM,
                              session_id.encode("utf-8"))
        await self._ws.send(packet)
        log.info(f"first packet sent (START_STREAM, session={session_id})")

    async def send_audio(self, pcm16: np.ndarray) -> None:
        payload = pcm16.astype("<i2").tobytes()
        packet = pack_packet(self._next_seq(), config.FLAGS.AUDIO, payload)
        await self._ws.send(packet)
        self.bytes_sent += len(payload)
        self.packets_sent += 1

    async def send_end_stream(self) -> None:
        packet = pack_packet(self._next_seq(), config.FLAGS.END_STREAM)
        await self._ws.send(packet)
        log.info("STREAM ended (END_STREAM sent)")

    async def recv(self):
        """Await the next message from the server (e.g. a transcription)."""
        return await self._ws.recv()
