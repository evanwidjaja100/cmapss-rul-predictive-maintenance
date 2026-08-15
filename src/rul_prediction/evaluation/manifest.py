"""Reusable manifest-based evaluation for Methodology V2.

Every model is evaluated on exactly the rows of a fixed pseudo-test manifest:
one terminal prediction per (engine, cutoff). Predictions are returned in
manifest order, so metrics and NASA totals are always computed over the same
sample set across models and window settings.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def evaluate_manifest(
    manifest: pd.DataFrame,
    trajectories: dict[int, pd.DataFrame],
    predict_one: Callable,
    preprocess: Callable | None = None,
) -> np.ndarray:
    """Return one prediction per manifest row, in manifest order.

    Parameters
    ----------
    manifest : pseudo-test manifest DataFrame (engine_id, cutoff_cycle, ...).
    trajectories : {engine_id: engine trajectory DataFrame with a ``cycle`` column}.
        Only rows with cycle <= cutoff are ever passed downstream (no future
        observations are available at prediction time).
    predict_one : callable(history, cutoff_cycle) -> scalar prediction.
        ``history`` contains only rows up to and including the cutoff cycle.
    preprocess : optional callable(history) -> history, e.g. a train-only scaler.
    """
    predictions = []
    for row in manifest.itertuples(index=False):
        engine = int(row.engine_id)
        cutoff = int(row.cutoff_cycle)
        history = trajectories[engine]
        history = history[history["cycle"] <= cutoff]
        if preprocess is not None:
            history = preprocess(history)
        predictions.append(predict_one(history, cutoff))
    return np.asarray(predictions, dtype=float)