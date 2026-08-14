"""NASA / PHM asymmetric RUL scoring function.

Score per engine:
    d = predicted - actual
    d < 0 (predicted < actual, i.e. EARLY): exp(-d/13) - 1
    d >= 0 (predicted > actual, i.e. LATE): exp(d/10) - 1

Late predictions are penalised more heavily (10 vs 13 in the exponent),
reflecting that predicting failure too late is costlier than too early.
Lower is better; loss = 0 when predictions are exact.
"""

from __future__ import annotations

import numpy as np


def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    early = np.exp(-d / 13.0) - 1.0  # d < 0
    late = np.exp(d / 10.0) - 1.0  # d >= 0
    asymmetric = np.where(d < 0, early, late)
    return float(np.sum(asymmetric))