"""Tests for regression metrics and the NASA asymmetric score."""

import numpy as np

from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score


def test_rmse():
    y_true = np.array([0.0, 0.0, 0.0])
    y_pred = np.array([0.0, 0.0, 1.0])
    assert rmse(y_true, y_pred) == np.sqrt(1.0 / 3.0)


def test_mae():
    assert mae(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])) == 1.0 / 3.0


def test_r2_perfect_is_one():
    assert r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 1.0


def test_r2_known_value():
    # ss_res = 0.25, ss_tot = 2 -> 1 - 0.25/2 = 0.875
    assert r2(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.5])) == 0.875


def test_nasa_zero_for_exact():
    assert nasa_score(np.array([10.0, 20.0]), np.array([10.0, 20.0])) == 0.0


def test_nasa_score_single_late_and_early():
    # late: d = 12 - 10 = 2 -> exp(2/10) - 1
    # early: d = 8 - 10 = -2 -> exp(2/13) - 1
    expected = (np.exp(0.2) - 1.0) + (np.exp(2.0 / 13.0) - 1.0)
    y_true = np.array([10.0, 10.0])
    y_pred = np.array([12.0, 8.0])
    assert np.isclose(nasa_score(y_true, y_pred), expected)

    # late prediction dominates
    late = nasa_score(np.array([10.0]), np.array([11.0]))  # d=1 late
    early = nasa_score(np.array([10.0]), np.array([9.0]))  # d=-1 early
    assert late > early