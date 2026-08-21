"""Tests for history-only engineered features."""

import numpy as np

from rul_prediction.features.engineered_features import extract_features
import pytest

pytestmark = pytest.mark.unit


def test_feature_shape_and_names():
    X = np.zeros((4, 30, 3))  # 4 windows, 3 sensors
    Xf, names = extract_features(X, final_cycle=np.array([30.0, 40.0, 50.0, 60.0]))
    assert Xf.shape == (4, 8 * 3 + 1)
    assert len(names) == 8 * 3 + 1
    assert names[-1] == "engine_age"
    assert names[0] == "sensor_1_last"


def test_slope_of_linear_sensor_matches_rate():
    X = np.zeros((1, 30, 1))
    X[0, :, 0] = np.arange(30, dtype=float) * 2.0  # slope = 2.0
    Xf, names = extract_features(X)
    slope_idx = names.index("sensor_1_slope")
    assert np.isclose(Xf[0, slope_idx], 2.0)


def test_last5_and_last_mean():
    X = np.zeros((1, 30, 1))
    X[0, :, 0] = np.arange(30, dtype=float)  # 0..29
    Xf, names = extract_features(X)
    assert np.isclose(Xf[0, names.index("sensor_1_last")], 29.0)
    assert np.isclose(Xf[0, names.index("sensor_1_last5")], np.mean([25, 26, 27, 28, 29]))



def test_no_future_leakage_by_construction():
    # last features must equal the final cycle value, never a later value
    X = np.random.default_rng(0).normal(size=(2, 30, 4))
    Xf, names = extract_features(X)
    assert np.allclose(Xf[:, names.index("sensor_2_last")], X[:, -1, 1])
    assert np.allclose(Xf[:, names.index("sensor_4_last10")], X[:, -10:, 3].mean(axis=1))