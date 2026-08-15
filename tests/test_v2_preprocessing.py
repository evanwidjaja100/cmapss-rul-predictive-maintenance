"""Methodology V2 raw-RUL preprocessing tests (artifact-free)."""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.data.preprocessing import fit_scaler, transform
from rul_prediction.data.loader import sensor_columns
from rul_prediction.data.v2_preprocessing import (
    add_raw_rul,
    add_target,
    build_v2_train_sequences,
)
from rul_prediction.data.windows import build_window, window_mask


def _frame(lifetimes: dict[int, int], n_features: int = 3) -> pd.DataFrame:
    rows = []
    for engine, lifetime in lifetimes.items():
        for cycle in range(1, lifetime + 1):
            rows.append(
                {"engine_id": engine, "cycle": cycle,
                 **{f"sensor_{j + 1}": float(cycle) for j in range(n_features)}}
            )
    return pd.DataFrame(rows)


# ---- raw target definition --------------------------------------------------

def test_raw_rul_never_clipped():
    frame = add_raw_rul(_frame({1: 200, 2: 160}))
    assert frame["rul"].max() == 199  # max_cycle - cycle, no cap
    assert frame["rul"].max() > 45


def test_target_mode_explicit_raw_ignores_cap():
    raw = add_target(_frame({1: 200}), target_mode="raw", cap=45)
    capped = add_target(_frame({1: 200}), target_mode="capped", cap=45)
    assert raw["rul"].max() == 199
    assert capped["rul"].max() == 45
    assert (raw["rul"] > capped["rul"]).any()


def test_target_mode_requires_cap_when_capped():
    with pytest.raises(AssertionError):
        add_target(_frame({1: 100}), target_mode="capped", cap=None)


# ---- shared history builder -------------------------------------------------

def test_build_window_full_history():
    scaled = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)
    win, n_obs, padded = build_window(scaled, cutoff_cycle=40, window=30)
    assert win.shape == (30, 3)
    assert n_obs == 30 and padded is False
    np.testing.assert_array_equal(win, scaled[10:40])  # last 30 cycles up to cutoff


def test_build_window_pads_short_history():
    scaled = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)
    win, n_obs, padded = build_window(scaled, cutoff_cycle=10, window=30)
    assert n_obs == 10 and padded is True
    np.testing.assert_array_equal(win[:20], np.zeros((20, 3), dtype=np.float32))  # padding
    np.testing.assert_array_equal(win[20:], scaled[:10])  # observed cycles 1..10


def test_build_window_no_future_cycles():
    scaled = np.arange(50 * 3, dtype=np.float32).reshape(50, 3)
    for cutoff in (5, 15, 30, 45):
        win, n_obs, _ = build_window(scaled, cutoff_cycle=cutoff, window=30)
        observed = win[window_mask(n_obs, 30).astype(bool)]
        np.testing.assert_array_equal(observed, scaled[:cutoff][-n_obs:])


def test_build_window_cutoff_range_guarded():
    scaled = np.ones((10, 3), dtype=np.float32)
    with pytest.raises(AssertionError):
        build_window(scaled, cutoff_cycle=0, window=30)
    with pytest.raises(AssertionError):
        build_window(scaled, cutoff_cycle=11, window=30)


def test_window_mask():
    mask = window_mask(n_observed=10, window=30)
    assert mask[:20].sum() == 0 and mask[20:].sum() == 10


# ---- training sequences -----------------------------------------------------

def test_train_sequences_use_same_builder_as_inference():
    frame = add_raw_rul(_frame({1: 40}))
    scaler = fit_scaler(frame, sensor_columns(frame))
    scaled = transform(frame, sensor_columns(frame), scaler)
    X, y, ids, obs, masks = build_v2_train_sequences(scaled, frame["engine_id"].to_numpy(),
                                                     frame["rul"].to_numpy(), window=30)
    assert len(X) == 40  # one example per end-cycle 1..40, INCLUDING padded e < 30
    assert int(obs.sum()) == 40 * 30 - sum(30 - e for e in range(1, 30))  # padded total
    # e = 10 -> the same window the shared builder produces at inference time
    engine_block = scaled[frame["engine_id"].to_numpy() == 1]
    win, n_obs, padded = build_window(engine_block, cutoff_cycle=10, window=30)
    np.testing.assert_array_equal(X[9], win)
    assert obs[9] == n_obs and padded
    np.testing.assert_array_equal(masks[9], window_mask(n_obs, 30))


def test_train_sequences_raw_targets_uncapped():
    frame = add_raw_rul(_frame({1: 60, 2: 60}))
    scaled = transform(frame, sensor_columns(frame), fit_scaler(frame, sensor_columns(frame)))
    X, y, ids, obs, masks = build_v2_train_sequences(
        scaled, frame["engine_id"].to_numpy(), frame["rul"].to_numpy(), window=30)
    assert y.max() == 59  # raw RUL exceeds the legacy cap (45) in training data
    assert y[0] == 59  # end-cycle 1 -> rul 59 for a 60-cycle engine


def test_train_sequences_no_engine_boundary_crossing():
    frame = add_raw_rul(_frame({1: 40, 2: 40}))
    scaled = transform(frame, sensor_columns(frame), fit_scaler(frame, sensor_columns(frame)))
    X, y, ids, obs, masks = build_v2_train_sequences(
        scaled, frame["engine_id"].to_numpy(), frame["rul"].to_numpy(), window=30)
    assert set(np.unique(ids)) == {1, 2}
    for engine in (1, 2):
        block_rows = X[ids == engine]
        assert len(block_rows) == 40


# ---- train-only scaling -----------------------------------------------------

def test_scaler_fit_on_train_engines_only():
    frame = _frame({1: 50, 2: 50})  # engine 1 = train, engine 2 = validation
    train = frame[frame["engine_id"] == 1]
    val = frame[frame["engine_id"] == 2].copy()
    val["sensor_1"] = 99999.0  # poison validation row

    scaler = fit_scaler(train, sensor_columns(train))
    mean_before = scaler.mean_[0]
    transform(val, sensor_columns(frame), scaler)  # application must not mutate scaler
    assert scaler.mean_[0] == mean_before
    assert scaler.mean_[0] != pytest.approx(99999.0)


def test_partition_leakage_guard_in_script_contract():
    # scaler fitted on train rows; validation/calibration rows transformed with
    # the same scaler object - verify no train/validation engine overlap exists
    # in the V2 split contract (disjointness is asserted by splitting module).
    from rul_prediction.data.splitting import split_engine_ids_v2
    train, validation, calibration = split_engine_ids_v2(set(range(1, 101)), seed=42)
    assert train.isdisjoint(validation) and train.isdisjoint(calibration)
    assert validation.isdisjoint(calibration)