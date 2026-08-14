"""History-only feature engineering for non-temporal (classical) RUL models.

Features are computed per window from data up to and including the window's
final cycle (never future information): current (last) value, window mean/std/
min/max, linear slope over the window, and means of the last 5 / last 10
cycles. Engine age (final cycle) is added when provided.
"""

from __future__ import annotations

import numpy as np

WINDOW = 30


def _slopes(X: np.ndarray) -> np.ndarray:
    """Linear slope (least squares) per sensor over window; X: (n, W, F)."""
    n, w, f = X.shape
    t = np.arange(w, dtype=float)
    t_c = t - t.mean()
    denom = np.sum(t_c**2)
    # mean over window per sensor
    mean = X.mean(axis=1)  # (n, F)
    t_c3 = t_c[:, None]  # (W, 1) -> broadcast with (n,W,F)
    num = np.sum(t_c[None, :, None] * (X - mean[:, None, :]), axis=1)  # (n, F)
    return num / denom


def extract_features(X: np.ndarray, final_cycle: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """Return (feature_matrix (n, n_feat), feature_names)."""
    n, w, f = X.shape
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mn = X.min(axis=1)
    mx = X.max(axis=1)
    slope = _slopes(X)
    last5 = X[:, -5:, :].mean(axis=1) if w >= 5 else last
    last10 = X[:, -10:, :].mean(axis=1) if w >= 10 else last

    feats = [last, mean, std, mn, mx, slope, last5, last10]
    names = ["last", "mean", "std", "min", "max", "slope", "last5", "last10"]

    parts = [f.reshape(n, -1) for f in feats]
    out_names = []
    for ni, name in enumerate(names):
        for j in range(f):
            out_names.append(f"sensor_{j + 1}_{name}")

    if final_cycle is not None:
        parts.append(np.asarray(final_cycle, dtype=float).reshape(-1, 1))
        out_names.append("engine_age")

    return np.hstack(parts).astype(np.float32), out_names