"""
benchmarks/latency_test.py

Analyzes latency across the T1-T6 pipeline checkpoints:
    T1 keyword detection timestamp
    T2 streaming started
    T3 first packet sent
    T4 server received first packet
    T5 ASR processing started
    T6 transcription completed

edge_simulator/main.py logs edge-side timings (T1-T3) live; the
server logs T4-T6 per-utterance in its `latencies_ms` response
field. This script runs N live wake-word cycles against a running
server + edge simulator pair is impractical to automate headlessly,
so instead it operates in two modes:

  1. `--live N`   drive N synthetic utterances through the ASR server
                  directly (bypassing the mic/KWS stages) to measure
                  server-side latency (T4-T6) under load.
  2. `--csv path` summarize a CSV of already-collected latency rows
                  (columns: edge_trigger_latency_ms,network_latency_ms,
                  wake_to_server_ms,asr_latency_ms,total_latency_ms).

Usage:
    python benchmarks/latency_test.py --live 10
    python benchmarks/latency_test.py --csv benchmarks/results/latency.csv
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from edge_simulator.streaming import StreamingClient  # noqa: E402

log = get_logger("LATENCY")

RESULTS_CSV = config.BENCHMARK_RESULTS_DIR / "latency.csv"


async def _run_one_utterance(client: StreamingClient, duration_s: float = 1.5) -> dict:
    session_id = f"bench-{int(time.time() * 1000)}"

    t2 = time.monotonic()
    await client.send_start_stream(session_id)
    t3 = time.monotonic()

    # Synthetic PCM16 "speech" — silence is fine, we're measuring
    # transport + ASR wall time, not transcription quality.
    n_frames = int(duration_s * 1000 / config.FRAME_MS)
    for _ in range(n_frames):
        frame = (np.random.randn(config.FRAME_SAMPLES) * 500).astype(np.int16)
        await client.send_audio(frame)

    await client.send_end_stream()

    response_raw = await client.recv()
    t_response = time.monotonic()

    import json
    response = json.loads(response_raw)
    latencies = response.get("latencies_ms", {})

    return {
        "edge_trigger_latency_ms": (t2 - t2) * 1000,  # 0 in this synthetic harness
        "network_latency_ms": (t3 - t2) * 1000,
        "wake_to_server_ms": latencies.get("wake_to_server_ms", 0.0),
        "asr_latency_ms": latencies.get("asr_latency_ms", 0.0),
        "total_latency_ms": (t_response - t2) * 1000,
    }


async def run_live(n: int) -> list[dict]:
    client = StreamingClient()
    await client.connect()
    rows = []
    for i in range(n):
        row = await _run_one_utterance(client)
        rows.append(row)
        log.info(f"[{i + 1}/{n}] total_latency={row['total_latency_ms']:.1f}ms "
                  f"asr={row['asr_latency_ms']:.1f}ms")
    await client.close()
    return rows


def summarize(rows: list[dict]) -> None:
    if not rows:
        log.info("no rows to summarize")
        return
    fields = list(rows[0].keys())
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"saved {len(rows)} rows to {RESULTS_CSV}")

    for field in fields:
        values = [r[field] for r in rows]
        log.info(f"{field}: mean={statistics.mean(values):.1f}ms "
                  f"p50={statistics.median(values):.1f}ms "
                  f"max={max(values):.1f}ms")


def summarize_csv(path: Path) -> None:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = [{k: float(v) for k, v in row.items()} for row in reader]
    summarize(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live", type=int, default=None, help="Run N synthetic utterances against a live server.")
    parser.add_argument("--csv", type=str, default=None, help="Summarize an existing CSV instead of running live.")
    args = parser.parse_args()

    if args.csv:
        summarize_csv(Path(args.csv))
    elif args.live:
        rows = asyncio.run(run_live(args.live))
        summarize(rows)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
