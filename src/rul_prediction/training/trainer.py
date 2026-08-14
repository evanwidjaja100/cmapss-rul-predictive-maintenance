"""Deterministic training utilities for sequence models."""

from __future__ import annotations

import numpy as np


def set_seed(seed: int = 42) -> None:
    import tensorflow as tf

    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_sequence_model(model, X_train, y_train, X_val, y_val,
                         batch_size: int = 128, epochs: int = 40, callbacks=None, verbose=1):
    """Fit on training sequences, monitor the engine-disjoint validation partition."""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose,
    )
    return history