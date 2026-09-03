# Estimated ESP32-S3 memory budget

Target: **< 256 KB working RAM**, **< 10% idle CPU**, running the KWS
stage only (ASR stays on the server). These are estimates based on
the current model/buffer sizes in this repo; measure on real hardware
before treating them as final.

| Component                         | Estimated RAM | Notes |
|------------------------------------|---------------|-------|
| TFLite Micro tensor arena          | ~40-60 KB     | Sized to the DS-CNN in `training/model.py`; a `tflite::MicroInterpreter` arena must fit all intermediate activation tensors. Measure exactly via `interpreter.arena_used_bytes()` once ported — this is the single biggest line item and the first thing to shrink if over budget (reduce channel counts in `model.py`). |
| MFCC computation buffers           | ~2-4 KB       | STFT frame buffer (30ms window = 480 samples * 4 bytes float, or less with fixed-point) + mel filterbank output (40 mel bins) + output feature buffer (49x13 floats = ~2.5KB, or ~1.3KB if features are quantized to int8 before feeding the model). |
| Audio ring buffer (pre-roll)       | ~9.6 KB       | `PREROLL_MS=300` * `SAMPLE_RATE=16000` / 1000 * 2 bytes (int16) = 9,600 bytes. See `edge_simulator/ring_buffer.py`. |
| I2S DMA buffers                    | ~2-4 KB       | Typically 2-4 buffers of `FRAME_MS=20ms` * 16kHz * 2 bytes = 640 bytes each; a few buffers for double/triple buffering. |
| Network buffers (WebSocket TX)     | ~4-8 KB       | TLS/WebSocket library overhead + one outgoing audio packet (`FRAME_SAMPLES` * 2 bytes + 17-byte header ~= 657 bytes) times a small send queue. |
| Application state                  | ~1-2 KB       | KWS state machine, VAD state, session ID, smoothing score deque, misc globals. |
| **Estimated total**                | **~60-90 KB** | Comfortably under the 256KB target, leaving headroom for the WiFi/TCP/TLS stack (which on ESP-IDF typically reserves tens of KB of its own, separate from application RAM above). |

## Idle CPU (<10%)

The energy gate (`config.RMS_GATE_THRESHOLD`, implemented in
`edge_simulator/main.py`'s inference-gating logic and mirrored in
`edge_simulator/vad.py`'s `rms()` helper) is the primary mechanism
for this: cheap RMS computation runs on every frame, but the
comparatively expensive MFCC + TFLite Micro inference only runs
every `INFERENCE_INTERVAL_MS` (100ms default) *and* only when the
RMS gate passes. In near-silent rooms, inference essentially never
runs, keeping the CPU duty cycle low. Tune `RMS_GATE_THRESHOLD` per
deployment environment/microphone.

## Where these numbers come from

These are architectural estimates, not measurements from real
ESP32-S3 hardware — this repo currently targets a laptop-based
simulator (see `docs/esp32_mapping.md`). Once ported, replace this
table with actual `heap_caps_get_free_size()` / tensor-arena
measurements from the device.
