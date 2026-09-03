"""
training/train.py

End-to-end training: loads the speaker-split manifest (run
preprocess.py first), builds a tf.data pipeline with on-the-fly
augmentation + MFCC extraction, trains model.py's DS-CNN, and saves
the best checkpoint plus training metrics for the dashboard.

Usage:
    python training/preprocess.py     # writes dataset/manifest.json
    python training/train.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from logging_utils import get_logger  # noqa: E402
from training.preprocess import preprocess_waveform  # noqa: E402
from training.augmentation import augment_waveform  # noqa: E402
from training.features import waveform_to_mfcc_numpy  # noqa: E402
from training.model import build_model, summarize  # noqa: E402

log = get_logger("TRAIN")

LABEL_TO_INDEX = {label: i for i, label in enumerate(config.LABELS)}


def _load_manifest() -> dict:
    manifest_path = config.DATASET_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — run `python training/preprocess.py` first."
        )
    with open(manifest_path) as f:
        return json.load(f)


def _make_example(path: str, label: str, augment: bool, rng: np.random.RandomState):
    waveform = preprocess_waveform(path)
    if augment:
        waveform = augment_waveform(waveform, rng=rng)
    features = waveform_to_mfcc_numpy(waveform)
    return features.astype(np.float32), LABEL_TO_INDEX[label]


def _build_dataset(entries: list[dict], augment: bool, shuffle: bool) -> tf.data.Dataset:
    paths = [e["path"] for e in entries]
    labels = [e["label"] for e in entries]
    rng = np.random.RandomState(1234)

    def generator():
        indices = np.arange(len(paths))
        if shuffle:
            rng.shuffle(indices)
        for i in indices:
            yield _make_example(paths[i], labels[i], augment, rng)

    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=config.FEATURE_SHAPE, dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, max(1, len(paths))))
    ds = ds.batch(config.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    manifest = _load_manifest()
    for split in ("train", "val", "test"):
        if not manifest.get(split):
            log.info(f"WARNING: split '{split}' is empty — results will be unreliable "
                      f"until more data / speakers are collected.")

    train_ds = _build_dataset(manifest["train"], augment=True, shuffle=True)
    val_entries = manifest["val"] or manifest["train"][:max(1, len(manifest["train"]) // 10)]
    val_ds = _build_dataset(val_entries, augment=False, shuffle=False)

    model = build_model()
    summarize(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    checkpoint_path = config.MODELS_DIR / "kws_checkpoint.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path), monitor="val_loss", save_best_only=True,
        ),
    ]

    start = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed = time.time() - start
    log.info(f"training finished in {elapsed:.1f}s")

    # Save FP32 SavedModel + .h5-style keras model for quantize.py
    fp32_keras_path = config.MODELS_DIR / "kws_fp32.keras"
    model.save(fp32_keras_path)
    log.info(f"saved trained model to {fp32_keras_path}")

    # Persist metrics history for the dashboard.
    metrics = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    metrics["elapsed_seconds"] = elapsed
    metrics["train_samples"] = len(manifest["train"])
    metrics["val_samples"] = len(val_entries)
    metrics_path = config.MODELS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"saved training metrics to {metrics_path}")


if __name__ == "__main__":
    main()
