"""FD004 operating-condition preprocessing (Methodology M2, issue R13).

FD004 engines operate under several regimes (setting_2 drifts in bands,
setting_3 in {60, 100}; engines switch regime mid-life). A single global
scaler mixes regimes and was diagnosed as the cause of the M1-11 collapse
to a constant prediction.

M2 variants:
    A  global scaler, 21 sensor inputs (reproduces M1-11 baseline)
    B  global scaler on sensors + settings, 24 inputs
    C  per-regime scalers (KMeans k=6 on settings), 21 inputs
    D  C + settings features + one-hot regime, 30 inputs

Leakage rules (non-negotiable):
    - KMeans and every scaler fit on DEVELOPMENT-TRAINING rows only;
    - at inference, regime assignment uses only the fitted KMeans;
    - settings used for clustering are operating conditions, not labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from rul_prediction.data.m1_preprocessing import SENSOR_COLUMNS

SETTING_COLUMNS = ["setting_1", "setting_2", "setting_3"]


def fit_condition_models(frame: pd.DataFrame, engine_ids, k: int = 6, seed: int = 42,
                         n_init: int = 10):
    """KMeans (on settings) + per-cluster sensor scalers + settings scaler, fit on `engine_ids` rows only."""
    rows = frame[frame["engine_id"].isin(engine_ids)].sort_values(["engine_id", "cycle"])
    settings = rows[SETTING_COLUMNS].to_numpy(dtype=float)
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=n_init).fit(settings)
    labels = kmeans.predict(settings)
    sensors = rows[SENSOR_COLUMNS].to_numpy(dtype=float)
    cluster_scalers = {}
    for c in range(k):
        cluster_scalers[int(c)] = StandardScaler().fit(sensors[labels == c])
    settings_scaler = StandardScaler().fit(settings)
    return kmeans, cluster_scalers, settings_scaler


def condition_feature_matrix(frame: pd.DataFrame, kmeans, cluster_scalers,
                             settings_scaler, with_settings: bool = True,
                             with_regime: bool = True) -> np.ndarray:
    """Per-row feature matrix in `frame` row order.

    sensors: scaled by the row's regime cluster scaler (row-wise regime
    scaling - a window may span several regimes). Optionally followed by the
    scaled settings and the regime one-hot.
    """
    rows = frame.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    sensors = rows[SENSOR_COLUMNS].to_numpy(dtype=float)
    labels = kmeans.predict(rows[SETTING_COLUMNS].to_numpy(dtype=float))
    scaled_sensors = np.stack([
        cluster_scalers[int(c)].transform(sensors[i:i + 1])[0]
        for i, c in enumerate(labels)
    ]).astype(np.float32)
    parts = [scaled_sensors]
    if with_settings:
        parts.append(settings_scaler.transform(
            rows[SETTING_COLUMNS].to_numpy(dtype=float)).astype(np.float32))
    if with_regime:
        parts.append(np.eye(kmeans.n_clusters, dtype=np.float32)[labels])
    return np.hstack(parts), labels, rows["engine_id"].to_numpy()