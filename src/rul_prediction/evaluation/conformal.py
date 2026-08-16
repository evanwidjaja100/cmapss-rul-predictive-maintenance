"""Split-conformal helpers for Methodology V2 uncertainty calibration.

Inductive conformal prediction: nonconformity scores (absolute residuals) are
computed on the fixed calibration manifest, and the prediction interval is
``y_hat +/- q`` where ``q`` is the finite-sample quantile

    q = the ceil((n + 1) * (1 - alpha))-th smallest score  (1-indexed)

which guarantees marginal coverage >= 1 - alpha on exchangeable data
(the calibration set itself is covered by construction; held-out coverage is
empirical and must be measured, see scripts/calibrate_v2_conformal.py).
"""

from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample ICP quantile of absolute residuals (alpha in (0, 1))."""
    assert 0 < alpha < 1
    scores = np.asarray(scores, dtype=float)
    assert scores.ndim == 1 and scores.size > 0
    n = scores.size
    level = (n + 1) * (1 - alpha) / n
    return float(np.quantile(scores, level, method="higher"))


def interval_coverage(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    """Fraction of rows with |y_true - y_pred| <= q (empirical interval coverage)."""
    return float(np.mean(np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float)) <= q))