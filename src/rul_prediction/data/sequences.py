"""Sliding-window sequence generation.

Windows are created per engine and never cross engine boundaries. For a window
ending at cycle `e`, the target is the RUL of the final cycle in that window
(i.e. `rul` at row index `e - 1`).
"""

from __future__ import annotations

import numpy as np


def make_sequences(frame, feature_cols, window: int = 30):
    """Build (X, y, engine_ids) from a frame that already contains `rul`.

    Returns:
        X: float32 array, shape (n_windows, window, len(feature_cols))
        y: float32 array, shape (n_windows,)
        engine_ids: int array, shape (n_windows,) - engine of each window
    """
    Xs, ys, ids = [], [], []
    for engine, group in frame.groupby("engine_id"):
        if len(group) < window:
            continue
        features = group[feature_cols].to_numpy(dtype=np.float32)
        rul = group["rul"].to_numpy(dtype=np.float32)
        n = len(group)
        for end in range(window, n + 1):
            Xs.append(features[end - window : end])
            ys.append(rul[end - 1])
            ids.append(engine)
    if not Xs:
        return (
            np.empty((0, window, len(feature_cols)), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=int),
        )
    return (
        np.stack(Xs).astype(np.float32),
        np.array(ys, dtype=np.float32),
        np.array(ids, dtype=int),
    )