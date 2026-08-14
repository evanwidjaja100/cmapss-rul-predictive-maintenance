"""Classical (non-temporal) RUL baselines.

predict(engine age) + aggregation baselines use only training targets for their
reference value; per-window models fit well-established regressors.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class MeanBaseline:
    """Predict the training-set mean RUL for every query."""

    name = "mean"

    def fit(self, X, y):
        self.value = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.value)


def linear_regressor(seed: int = 42) -> LinearRegression:
    return LinearRegression()


def random_forest(seed: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=300, max_depth=None, n_jobs=-1, random_state=seed)