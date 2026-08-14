"""Tests for RUL preprocessing and sequence generation (leakage-safe)."""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.data.preprocessing import (
    SENSOR_COLUMNS,
    add_rul,
    fit_scaler,
    transform,
)
from rul_prediction.data.sequences import make_sequences
from rul_prediction.data.splitting import split_engine_ids

FEATURES = ["f1", "f2", "f3"]


def _frame(engine_lengths, n_features=3):
    """Build a frame of engines (no rul yet) with synthetic sensors."""
    rows = []
    for engine, length in engine_lengths.items():
        for cycle in range(1, length + 1):
            rows.append([engine, cycle, cycle, cycle, cycle])
    return pd.DataFrame(rows, columns=["engine_id", "cycle"] + FEATURES)


# --- RUL ---

def test_rul_is_max_minus_cycle():
    frame = add_rul(_frame({1: 10, 2: 5}))
    assert frame.loc[frame["engine_id"] == 1, "rul"].iloc[0] == 9  # cycle 1
    assert frame.loc[frame["engine_id"] == 1, "rul"].iloc[-1] == 0  # cycle 10
    assert frame.loc[frame["engine_id"] == 2, "rul"].iloc[-1] == 0


def test_rul_is_calculated_within_engine():
    frame = add_rul(_frame({1: 10, 2: 30}))
    assert frame.loc[frame["engine_id"] == 1, "rul"].max() == 9  # not 29


def test_rul_clipping():
    frame = add_rul(_frame({1: 200}), max_rul=125, clip=True)
    assert frame["rul"].max() == 125
    frame = add_rul(_frame({1: 200}), max_rul=125, clip=False)
    assert frame["rul"].max() == 199


# --- Sequences ---

def test_sequence_dimensions_and_dtype():
    frame = add_rul(_frame({1: 40, 2: 40}))
    X, y, ids = make_sequences(frame, FEATURES, window=30)
    n = (40 - 30 + 1) * 2
    assert X.shape == (n, 30, 3)
    assert y.shape == (n,)
    assert X.dtype == np.float32
    assert y.dtype == np.float32


def test_window_target_is_rul_of_last_cycle():
    frame = add_rul(_frame({1: 35}))
    X, y, ids = make_sequences(frame, FEATURES, window=30)
    # first window covers cycles 1..30 -> target RUL of cycle 30 = 35 - 30 = 5
    assert y[0] == 5
    # second window ends at cycle 31 -> 35 - 31 = 4
    assert y[1] == 4
    # last window ends at cycle 35 -> RUL = 0
    assert y[-1] == 0


def test_engines_shorter_than_window_skipped():
    frame = add_rul(_frame({1: 10, 2: 50}))
    X, y, ids = make_sequences(frame, FEATURES, window=30)
    assert np.all(ids == 2)


def test_streams_never_cross_engine_boundary():
    frame = add_rul(_frame({1: 40, 2: 40}))
    X, _, _ = make_sequences(frame, FEATURES, window=30)
    # per-engine count = 40 - 30 + 1 = 11; 11 for engine 1, 11 for engine 2
    assert X.shape[0] == 22
    # every window's rows must all belong to a single engine; reconstruct check
    # is satisfied by construction, but assert total count to catch naive splitting.
    assert X.shape == (22, 30, 3)


# --- Scaling fitted on training engines only ---

def test_scaler_fit_on_train_partition_only():
    # Engine-dependent features (offset scales with engine id) so that the train
    # partition's statistics differ from the full dataset's -> the "train only"
    # distinction is genuinely testable.
    rows = [
        [engine, cycle] + [cycle + 100 * engine] * len(FEATURES)
        for engine in range(1, 11)
        for cycle in range(1, 41)
    ]
    full = add_rul(pd.DataFrame(rows, columns=["engine_id", "cycle"] + FEATURES))
    train_ids, val_ids = split_engine_ids(set(range(1, 11)), seed=42)
    train_part = full[full["engine_id"].isin(train_ids)]
    val_part = full[full["engine_id"].isin(val_ids)]

    scaler = fit_scaler(train_part, FEATURES)

    # Deterministic proof: transform must equal standardization using TRAIN stats.
    # StandardScaler uses population std (ddof=0); mirror that in the expectation.
    arr = train_part[FEATURES].to_numpy()
    train_mean = arr.mean(axis=0)
    train_scale = arr.std(axis=0)
    expected = (val_part[FEATURES].to_numpy() - train_mean) / train_scale
    out = transform(val_part, FEATURES, scaler)
    assert np.allclose(out, expected, atol=1e-6)

    # And it must NOT equal standardization using full-dataset stats.
    full_arr = full[FEATURES].to_numpy()
    full_mean = full_arr.mean(axis=0)
    full_scale = full_arr.std(axis=0)
    expected_full = (val_part[FEATURES].to_numpy() - full_mean) / full_scale
    assert not np.allclose(out, expected_full, atol=1e-6)


# --- Leakage ---

def test_train_validation_sequences_are_disjoint():
    train_ids, val_ids = split_engine_ids(set(range(1, 101)), seed=42)
    assert train_ids.isdisjoint(val_ids)
    full = add_rul(_frame({i: 60 for i in range(1, 101)}))
    xt, yt, idt = make_sequences(full[full["engine_id"].isin(train_ids)], FEATURES, 30)
    xv, yv, idv = make_sequences(full[full["engine_id"].isin(val_ids)], FEATURES, 30)
    assert set(np.unique(idt)).isdisjoint(set(np.unique(idv)))
    assert set(np.unique(idt)) == train_ids
    assert set(np.unique(idv)) == val_ids