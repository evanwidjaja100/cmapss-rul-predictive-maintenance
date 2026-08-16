"""Serving core for the frozen V2 model (Methodology V2, Phase V2-9).

Wraps the frozen GRU w45 huber model (raw RUL target) with the identical
inference path used by the freeze and all V2 analyses (make_predictor +
train-only scaler + shared window builder) and adds the Phase V2-8 conformal
interval (alpha=0.1, q from reports/tables/v2_conformal_calibration.csv) and
the Phase V2-6 out-of-distribution flag (observed history shorter than the
training lifetime minimum -> measured 47.7% coverage, see reports/v2_conformal.md).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from keras import models as keras_models

from rul_prediction.benchmark.v2 import ROOT, load_v2_artifacts, make_predictor

WINDOW = 45
OOD_LIFETIME = 128  # train engine lifetime minimum (V2-6 finding)
ALPHA = 0.1


class V2Predictor:
    """One-terminal-prediction-per-engine serving wrapper."""

    def __init__(self, alpha: float = ALPHA, q_cycles: float | None = None) -> None:
        artifacts = load_v2_artifacts()
        self.scaler = artifacts["scaler"]
        self.model = keras_models.load_model(ROOT / "models" / "v2_frozen_gru_w45_huber.keras")
        self._predict_one = make_predictor("gru", self.model, self.scaler, WINDOW)
        if q_cycles is None:
            table = pd.read_csv(ROOT / "reports" / "tables" / "v2_conformal_calibration.csv")
            q_cycles = float(table.set_index("alpha").loc[alpha, "q_cycles"])
        self.q_cycles = float(q_cycles)
        self.alpha = alpha

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-engine terminal prediction on a C-MAPSS frame (engine_id, cycle, sensors...).

        Returns one row per engine with the raw-RUL prediction, the 90%
        conformal interval and the short-history OOD flag.
        """
        if not {"engine_id", "cycle"}.issubset(frame.columns):
            raise ValueError("frame must contain 'engine_id' and 'cycle' columns")
        rows = []
        for engine, g in frame.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
            history = g.reset_index(drop=True)
            cutoff = int(history["cycle"].iloc[-1])
            pred = float(self._predict_one(history, cutoff))
            n = int(len(history))
            rows.append({
                "engine_id": int(engine),
                "n_cycles": n,
                "prediction_raw_rul": round(pred, 2),
                "lo_90": round(pred - self.q_cycles, 2),
                "hi_90": round(pred + self.q_cycles, 2),
                "interval_width_90": round(2 * self.q_cycles, 2),
                "alarm_lower_bound": round(pred - self.q_cycles, 2),
                "ood_short_history": bool(n < OOD_LIFETIME),
            })
        return pd.DataFrame(rows)