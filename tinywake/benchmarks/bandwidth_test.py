"""
benchmarks/bandwidth_test.py

Measures streaming bandwidth for a given audio duration under the
current codec (PCM16 first; the packet/session accounting here is
codec-agnostic so IMA-ADPCM and Opus can be dropped in later by
changing only how bytes-per-sample is computed).

Usage:
    python benchmarks/bandwidth_test.py --seconds 10
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402

log = get_logger("BANDWIDTH")

# Bytes-per-sample for each codec this project plans to support.
# Only PCM16 is actually implemented in streaming.py today; ADPCM and
# Opus are placeholders for the projected bandwidth comparison.
CODEC_BYTES_PER_SAMPLE = {
    "PCM16": 2.0,
    "IMA-ADPCM": 0.5,   # 4 bits/sample
    "Opus": 0.25,       # ~ typical 16kbps voice encode approximation
}


def estimate_bandwidth(duration_s: float) -> dict:
    total_samples = int(duration_s * config.SAMPLE_RATE)
    packet_samples = config.FRAME_SAMPLES
    packet_count = -(-total_samples // packet_samples)  # ceil div

    header_bytes_per_packet = 17  # see streaming.py _HEADER_SIZE

    results = {}
    for codec, bytes_per_sample in CODEC_BYTES_PER_SAMPLE.items():
        payload_bytes = int(total_samples * bytes_per_sample)
        total_bytes = payload_bytes + packet_count * header_bytes_per_packet
        kbps = (total_bytes * 8 / 1000) / duration_s if duration_s > 0 else 0.0
        results[codec] = {
            "bytes_transmitted": total_bytes,
            "average_kbps": kbps,
            "packet_count": packet_count,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()

    log.info(f"audio duration: {args.seconds:.1f}s")
    results = estimate_bandwidth(args.seconds)
    for codec, stats in results.items():
        log.info(f"[{codec}] bytes={stats['bytes_transmitted']:,}  "
                  f"avg_kbps={stats['average_kbps']:.1f}  packets={stats['packet_count']}")

    log.info("Note: only PCM16 is currently implemented in streaming.py; "
              "ADPCM/Opus figures above are projected, not measured.")


if __name__ == "__main__":
    main()
