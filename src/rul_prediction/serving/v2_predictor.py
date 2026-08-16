"""Serving core for the frozen V2.1 model (Methodology V2.1).

Wraps the CV-selected GRU w45 huber model (raw RUL target) with the identical
inference path used by freeze and V2.1 analyses (make_predictor + scaler fit
on the 85 development engines + shared window builder) and the engine-level
conformal interval (alpha=0.1, q from reports/tables/v2_1_conformal_q.csv,
engine cluster of 15 calibration engines, n=15).

Terminology (V2_1_REPAIR_PLAN.md R1/R3): official C-MAPSS test trajectories
are truncated before failure; the number of observed cycles is an observed
history length, never a lifetime. There is no OOD classification here:

- ``history_is_padded``: objective - observed cycles < model window, the
  window is left-padded in the shared representation;
- ``short_history_risk_flag``: EMPIRICAL flag (observed < 128) derived from
  the V2.1 error analysis (overprediction concentrates there); it is not a
  distribution-shift (OOD) claim.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from keras import models as keras_models

from rul_prediction.benchmark.v2 import ROOT, make_predictor

WINDOW = 45
RISK_OBSERVED_CYCLES = 128  # empirical threshold from reports/v2_1_error_analysis.md
ALPHA = 0.1


class V2Predictor:
    """One-terminal-prediction-per-engine serving wrapper (V2.1)."""

    def __init__(self, alpha: float = ALPHA, q_cycles: float | None = None) -> None:
        from joblib import load as load_joblib

        self.model = keras_models.load_model(ROOT / "models" / "v2_1" / "fd001_gru_w45_huber.keras")
        self.scaler = load_joblib(ROOT / "models" / "v2_1" / "fd001_scaler.joblib")
        self._predict_one = make_predictor("gru", self.model, self.scaler, WINDOW)
        if q_cycles is None:
            table = pd.read_csv(ROOT / "reports" / "tables" / "v2_1_conformal_q.csv")
            q_cycles = float(table.set_index("alpha").loc[alpha, "q"])
        self.q_cycles = float(q_cycles)
        self.alpha = alpha

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-engine terminal prediction on a C-MAPSS frame (engine_id, cycle, sensors...).

        Returns one row per engine with the raw-RUL prediction, the 90%
        engine-cluster conformal interval and objective/empirical history flags.
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
                "n_cycles_observed": n,
                "history_is_padded": bool(n < WINDOW),
                "short_history_risk_flag": bool(n < RISK_OBSERVED_CYCLES),
                "prediction_raw_rul": round(pred, 2),
                "lo_90": round(pred - self.q_cycles, 2),
                "hi_90": round(pred + self.q_cycles, 2),
                "interval_width_90": round(2 * self.q_cycles, 2),
                "alarm_lower_bound": round(pred - self.q_cycles, 2),
            })
        return pd.DataFrame(rows)


def limited_history_warning(n_observed: int) -> str | None:
    """Exact warning text for short observed histories (None when full window)."""
    if n_observed >= WINDOW:
        return None
    padded = WINDOW - n_observed
    return (f"Limited observed history: only {n_observed} cycles observed; "
            f"window {WINDOW} -> {padded} timesteps padded.")