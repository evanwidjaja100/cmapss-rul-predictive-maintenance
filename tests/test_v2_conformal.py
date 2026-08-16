"""Tests for split-conformal helpers (Methodology V2, Phase V2-8)."""

import numpy as np

from rul_prediction.evaluation.conformal import conformal_quantile, interval_coverage


def test_quantile_is_ceil_order_statistic():
    scores = np.arange(1, 76, dtype=float)  # n = 75, k-th smallest = k
    assert conformal_quantile(scores, 0.1) == 69  # ceil((75+1)*0.9) = 69
    assert conformal_quantile(scores, 0.2) == 61  # ceil(76*0.8) = 61
    assert conformal_quantile(scores, 0.3) == 54  # ceil(76*0.7) = 54


def test_coverage_matches_quantile_definition():
    scores = np.arange(1, 76, dtype=float)
    q = conformal_quantile(scores, 0.1)
    y_true = scores
    y_pred = np.zeros_like(scores)
    assert interval_coverage(y_true, y_pred, q) == 69 / 75  # 69 of 75 scores <= q


def test_calibration_set_coverage_at_least_nominal():
    rng = np.random.default_rng(42)
    for alpha in (0.1, 0.2, 0.3):
        cal_scores = np.abs(rng.normal(0, 1, 200))
        q = conformal_quantile(cal_scores, alpha)
        assert interval_coverage(cal_scores, np.zeros_like(cal_scores), q) >= 1 - alpha


def test_iid_holdout_coverage_empirically_nominal():
    rng = np.random.default_rng(7)
    cal_scores = np.abs(rng.normal(0, 1, 500))
    holdout = np.abs(rng.normal(0, 1, 500))
    for alpha in (0.1, 0.2):
        q = conformal_quantile(cal_scores, alpha)
        cov = interval_coverage(holdout, np.zeros_like(holdout), q)
        assert cov >= 1 - alpha - 0.03  # finite-sample slack for n=500
        assert cov <= 1 - alpha + 0.03