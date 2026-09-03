# ESP32-S3 porting map

This project is built and validated entirely on a laptop first (per
the project's development strategy). This document tracks which
Python module corresponds to which future embedded component, so the
eventual ESP-IDF port is a translation exercise rather than a
redesign.

| Laptop module (Python)          | ESP32-S3 equivalent                          | Notes |
|----------------------------------|-----------------------------------------------|-------|
| `edge_simulator/microphone.py`   | I2S DMA driver (ESP-IDF `i2s_std` or `i2s_pdm`) | Same fixed-frame-size contract (`FRAME_MS` = 20ms); replace the bounded `queue.Queue` with a small DMA buffer ring (2-4 buffers). |
| `training/features.py`           | Embedded MFCC (e.g. `esp-dsp` FFT + fixed-point mel filterbank, or a hand-rolled fixed-point MFCC) | The `tf.signal` ops here define the *reference* math; the embedded version must match closely enough that the INT8 model (trained on these features) still performs well. Validate with `evaluate.py` against features extracted on-device before trusting field results. |
| `edge_simulator/kws_engine.py`   | TensorFlow Lite Micro `tflite::MicroInterpreter` running `kws_int8.tflite` | Same INT8 quantized model, same input/output tensor shapes. The state machine (`State` enum, smoothing, cooldown) is plain C-portable logic with no Python dependencies — copy the algorithm directly. |
| `edge_simulator/ring_buffer.py`  | Static circular buffer (fixed C array, no `malloc` after init) | Already written as a single pre-allocated NumPy array with a write pointer specifically to mirror this — see the class docstring. |
| `edge_simulator/streaming.py`    | ESP-IDF WebSocket client (`esp_websocket_client`) | The binary packet header (`struct.pack("<IdIB", ...)`) is a fixed-layout little-endian struct — define the identical `struct` in C and it round-trips with `server/audio_receiver.py` unchanged. |
| `edge_simulator/vad.py`          | Embedded VAD (same RMS-threshold algorithm, integer math) | No ML model involved — trivially portable; keep using `int32` accumulators instead of Python's arbitrary-precision floats for the sum-of-squares. |

## Not ported to ESP32 (stays server-side)

- `server/*.py` (FastAPI server, session tracking, `faster-whisper` ASR) — runs on
  a normal machine/server that the ESP32 connects to over WiFi; never runs
  on the microcontroller.
- `training/*.py` — training happens on a laptop/workstation; only the
  exported `kws_int8.tflite` (via `training/export_esp32.py` ->
  `model_data.cc/.h`) ships to the device.
- `dashboard/` — runs in a browser, connects to the server, not the device.

## Model export

`training/export_esp32.py` converts `models/kws_int8.tflite` into
`model_data.cc` / `model_data.h` (a C byte array + length), ready to
be compiled into an ESP-IDF project and passed to
`tflite::GetModel()`.
