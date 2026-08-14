"""Keras callbacks used by sequence models. Validation = engine-disjoint partition (never official test)."""

from __future__ import annotations

from pathlib import Path


def build_callbacks(checkpoint_path: str | Path, patience: int = 8, min_lr: float = 1e-6):
    from tensorflow.keras.callbacks import (
        EarlyStopping,
        ModelCheckpoint,
        ReduceLROnPlateau,
    )

    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=min_lr, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True),
    ]