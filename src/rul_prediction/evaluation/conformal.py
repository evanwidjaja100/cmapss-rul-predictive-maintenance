"""Split-conformal helpers for Methodology V2 / V2.1 uncertainty calibration.

Inductive conformal prediction: nonconformity scores (absolute residuals) are
computed on a fixed calibration set, and the prediction interval is
``y_hat +/- q`` where ``q`` is the finite-sample quantile.

Methodology V2.1 (see V2_1_REPAIR_PLAN.md, R9/R10) calibrates at the ENGINE
level: one score per calibration engine, taken as the maximum absolute error
across its five fixed lifecycle checkpoints. The finite-sample quantile index
is ``k = ceil((n + 1) * (1 - alpha))`` clamped to ``1 <= k <= n`` (1-indexed
ordered scores), which yields simultaneous coverage >= 1 - alpha over the
checkpoint scheme for exchangeable engines.
"""

from __future__ import annotations

import math

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample ICP quantile of scores (alpha in (0, 1)).

    V2 behavior preserved for compatibility: level = (n+1)*(1-alpha)/n with
    the 'higher' interpolation. V2.1 code should use
    ``finite_sample_quantile_index`` + ordered scores instead.
    """
    assert 0 < alpha < 1
    scores = np.asarray(scores, dtype=float)
    assert scores.ndim == 1 and scores.size > 0
    n = scores.size
    level = (n + 1) * (1 - alpha) / n
    return float(np.quantile(scores, level, method="higher"))


def finite_sample_quantile_index(n: int, alpha: float) -> int:
    """1-indexed finite-sample quantile index k = ceil((n+1)(1-alpha)), clamped to [1, n]."""
    assert 0 < alpha < 1
    assert n >= 1
    k = math.ceil((n + 1) * (1 - alpha))
    return int(min(max(k, 1), n))


def engine_cluster_scores(errors: np.ndarray) -> np.ndarray:
    """Engine-level nonconformity scores: max |error| over the checkpoint rows.

    errors: (n_engines, n_checkpoints) -> (n_engines,) max absolute error per engine.
    """
    errors = np.asarray(errors, dtype=float)
    assert errors.ndim == 2, "expected (n_engines, n_checkpoints)"
    return np.max(np.abs(errors), axis=1)


def quantile_from_index(ordered_scores: np.ndarray, k: int) -> float:
    """q = k-th smallest score (1-indexed), k pre-clamped via finite_sample_quantile_index."""
    ordered = np.sort(np.asarray(ordered_scores, dtype=float))
    assert ordered.size >= k >= 1
    return float(ordered[k - 1])


def interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Fraction of rows with |y_true - y_pred| <= q (empirical interval coverage)."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)) <= q))