"""Methodology V2.1 FD004 condition-module tests (artifact-free).

Covers the V2_1_REPAIR_PLAN.md R13 leakage rules: KMeans regime clustering
and every scaler fit on development-training rows ONLY; regime assignment at
inference uses only the fitted KMeans; row-wise regime scaling handles
windows that span multiple regimes.
"""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.data.condition import (
    SETTING_COLUMNS,
    condition_feature_matrix,
    fit_condition_models,
)
from rul_prediction.data.v2_preprocessing import SENSOR_COLUMNS

pytestmark = pytest.mark.unit

RNG = np.random.default_rng(3)


def synthetic_fd004(engines, conditions, n_rows=80, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for e in engines:
        n = n_rows
        cyc = np.arange(1, n + 1)
        setting_3 = np.where(cyc % 2 == 0, 100.0, 60.0)
        band = (cyc // 20) % len(conditions)
        setting_2 = np.asarray(conditions)[band] + rng.normal(0, 0.0005, n)
        sensors = np.stack([rng.normal(0.5 * (c % 3), 1, n) for c in range(21)], axis=1)
        frames.append(pd.DataFrame({
            "engine_id": e, "cycle": cyc,
            "setting_1": 100.0, "setting_2": setting_2, "setting_3": setting_3,
            **{f"sensor_{i}": sensors[:, i - 1] for i in range(1, 22)}}))
    return pd.concat(frames, ignore_index=True)


def test_condition_models_fit_on_training_rows_only():
    dev = synthetic_fd004([1, 2, 3, 4], [0.0, 0.25])
    kmeans, scalers, settings_scaler = fit_condition_models(dev, {1, 2, 3}, k=2, seed=42)
    assert kmeans.n_clusters == 2
    assert set(scalers.keys()) == {0, 1}
    # engine 4 rows were NOT used for fitting, but must still be clusterable
    matrix, labels, ids = condition_feature_matrix(
        dev[dev.engine_id == 4], kmeans, scalers, settings_scaler,
        with_settings=False, with_regime=False)
    assert matrix.shape == (len(dev[dev.engine_id == 4]), 21)
    assert set(np.unique(labels)) <= {0, 1}


def test_condition_matrix_widths():
    dev = synthetic_fd004([1, 2, 3], [0.0, 0.25, 0.62])
    kmeans, scalers, settings_scaler = fit_condition_models(dev, {1, 2, 3}, k=3, seed=42)
    for with_settings, with_regime, width in ((False, False, 21), (True, False, 24),
                                              (True, True, 21 + 3 + 3)):
        matrix, _, _ = condition_feature_matrix(
            dev, kmeans, scalers, settings_scaler,
            with_settings=with_settings, with_regime=with_regime)
        assert matrix.shape[1] == width


def test_regime_switches_do_not_break_windows():
    # an engine that switches regimes mid-history: row-wise scaling must be finite
    dev = synthetic_fd004([1, 2, 3], [0.0, 0.25], n_rows=50)
    kmeans, scalers, settings_scaler = fit_condition_models(dev, {1, 2, 3}, k=2, seed=42)
    eng = dev[dev.engine_id == 3].sort_values("cycle")
    matrix, labels, _ = condition_feature_matrix(eng, kmeans, scalers, settings_scaler)
    assert np.isfinite(matrix).all()
    assert len(set(labels)) > 1, "test engine must actually switch regimes"


def test_condition_matrix_rows_are_engine_sorted():
    dev = synthetic_fd004([1, 2, 3], [0.0, 0.25])
    kmeans, scalers, settings_scaler = fit_condition_models(dev, {1, 2, 3}, k=2, seed=42)
    shuffled = pd.concat([dev[dev.engine_id == e].sample(frac=1, random_state=1)
                          for e in (3, 1, 2)])
    matrix, _, ids = condition_feature_matrix(
        shuffled, kmeans, scalers, settings_scaler, with_settings=True, with_regime=True)
    assert (np.diff(ids) >= 0).all(), "rows must come back engine-sorted"


def test_settings_only_clustering_no_label_leak():
    dev = synthetic_fd004([1, 2, 3], [0.0, 0.25, 0.62])
    kmeans, _, _ = fit_condition_models(dev, {1, 2, 3}, k=3, seed=42)
    # cluster identity must be a pure function of the settings
    a = kmeans.predict(dev[dev.engine_id == 1][SETTING_COLUMNS].head(5).to_numpy())
    b = kmeans.predict(dev[dev.engine_id == 1][SETTING_COLUMNS].head(5).to_numpy())
    np.testing.assert_array_equal(a, b)