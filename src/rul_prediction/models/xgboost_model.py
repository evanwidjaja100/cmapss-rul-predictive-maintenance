"""XGBoost regressor wrapper for RUL prediction."""

from __future__ import annotations

from xgboost import XGBRegressor


def xgboost_regressor(seed: int = 42) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=seed,
        early_stopping_rounds=30,
    )