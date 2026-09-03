"""
training/model.py

A very small Depthwise-Separable CNN (DS-CNN) for keyword spotting,
sized to comfortably fit an INT8 TFLite Micro deployment on an
ESP32-S3 (target: FP32 < 500KB, INT8 < 100KB).

Input:  (49, 13, 1)  — MFCC features, see features.py / config.FEATURE_SHAPE
Output: (3,)         — softmax over [silence, unknown, hey_nova]
"""
from __future__ import annotations
import sys
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def build_model(input_shape=config.FEATURE_SHAPE, num_classes: int = len(config.LABELS)) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="mfcc_input")

    # Standard conv stem — small channel count keeps params low.
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same",
                       use_bias=False, name="conv_stem")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    # DS-CNN block 1
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(1, 1), padding="same",
                                use_bias=False, name="dwconv_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.Conv2D(24, kernel_size=(1, 1), padding="same",
                       use_bias=False, name="pwconv_1")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    # DS-CNN block 2
    x = layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), padding="same",
                                use_bias=False, name="dwconv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.Conv2D(32, kernel_size=(1, 1), padding="same",
                       use_bias=False, name="pwconv_2")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="logits")(x)

    return models.Model(inputs, outputs, name="tinywake_dscnn")


def estimate_tflite_size_bytes(model: tf.keras.Model) -> int:
    """Rough FP32 size estimate: 4 bytes per parameter plus small overhead."""
    return model.count_params() * 4 + 2048


def summarize(model: tf.keras.Model) -> None:
    total_params = model.count_params()
    est_fp32_kb = estimate_tflite_size_bytes(model) / 1024
    print("=" * 60)
    print(f"Model: {model.name}")
    print(f"Input shape:  {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {total_params:,}")
    print(f"Estimated FP32 size: ~{est_fp32_kb:.1f} KB (target < 500 KB)")
    print(f"Estimated INT8 size: ~{est_fp32_kb / 4:.1f} KB (target < 100 KB)")
    print("=" * 60)


if __name__ == "__main__":
    m = build_model()
    m.summary()
    summarize(m)
