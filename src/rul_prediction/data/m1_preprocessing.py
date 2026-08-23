"""Methodology M1 raw-RUL preprocessing.

Primary target: ``raw_rul = max_cycle(engine) - cycle`` — NEVER clipped.
Target mode is always explicit (``target_mode: raw | capped``), never inferred
from a directory or file name.

Leakage rules (unchanged, non-negotiable):
    - scaler fits on TRAINING ENGINES ONLY, applied unchanged to all partitions;
    - targets are computed per engine from its own lifetime (no cross-engine info);
    - sequences never cross an engine boundary;
    - no future cycles enter any window (see ``data.windows.build_window``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .windows import build_window, window_mask

TARGET_MODES = ("raw", "capped")

SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]


def add_raw_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `rul = max_cycle(engine) - cycle` with NO clipping."""
    frame = frame.copy()
    frame["rul"] = frame.groupby("engine_id")["cycle"].transform("max") - frame["cycle"]
    return frame


def add_target(frame: pd.DataFrame, target_mode: str = "raw", cap: int | None = None) -> pd.DataFrame:
    """Explicit target construction. raw: no cap. capped: clip at `cap` (secondary experiment)."""
    assert target_mode in TARGET_MODES, f"target_mode must be one of {TARGET_MODES}"
    frame = add_raw_rul(frame)
    if target_mode == "capped":
        assert cap is not None, "cap is required when target_mode='capped'"
        frame["rul"] = frame["rul"].clip(upper=cap)
    return frame


def build_m1_train_sequences(
    scaled: np.ndarray, engine_ids: np.ndarray, rul: np.ndarray, window: int
):
    """Training representation for Methodology M1 sequence models.

    For every engine, one example per end-cycle ``e`` in 1..n (n = lifetime),
    built with the SAME left-padding implementation as inference
    (``data.windows.build_window``), so short-history examples are present in
    training exactly as they will be at inference time.

    Returns (X, y, engine_ids, n_observed, masks):
        X         float32 (n_seq, window, n_features)
        y         float32 (n_seq,) raw (or capped) RUL at the window end
        engine_ids int32   (n_seq,)
        n_observed int32   (n_seq,) observed cycles per window
        masks      float32 (n_seq, window) 1=observed, 0=padded
    """
    Xs, ys, ids, obs, masks = [], [], [], [], []
    for engine in np.unique(engine_ids):
        block = scaled[engine_ids == engine]
        rul_block = rul[engine_ids == engine]
        for e in range(1, len(block) + 1):
            window_arr, n_observed, _ = build_window(block, e, window)
            Xs.append(window_arr)
            ys.append(rul_block[e - 1])
            ids.append(engine)
            obs.append(n_observed)
            masks.append(window_mask(n_observed, window))
    if not Xs:
        return (
            np.empty((0, window, scaled.shape[1]), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=int),
            np.empty(0, dtype=int),
            np.empty((0, window), dtype=np.float32),
        )
    return (
        np.stack(Xs).astype(np.float32),
        np.array(ys, dtype=np.float32),
        np.array(ids, dtype=int),
        np.array(obs, dtype=int),
        np.stack(masks).astype(np.float32),
    )


def target_distribution(rul: np.ndarray) -> dict:
    """Raw-RUL distribution diagnostics for the training partition (never model selection)."""
    return {
        "min": float(rul.min()),
        "max": float(rul.max()),
        "mean": float(rul.mean()),
        "median": float(np.median(rul)),
        "n_above_45": int((rul > 45).sum()),
        "n_zero": int((rul == 0).sum()),
    }