"""Serving core for the frozen V2.2 model (Methodology V2.2).

Wraps the deployment model chosen by the pre-specified V2.2 selection policy
(configs/final_model_v2_2_fd001.yaml) with the identical inference path used
by the freeze (scaler fit on the 85 development engines + shared window
builder) and the engine-cluster conformal interval. The interval ``q`` is read
from the TRACKED deployment config (configs/deployment_v2_2_fd001.yaml); the
experiment CSV (experiments/v2_2/fd001_conformal_quantiles.csv) remains the
audit source and is cross-checked when present.

Terminology (V2_2_REPAIR_PLAN.md): official C-MAPSS test trajectories are
truncated before failure; the number of observed cycles is an observed history
length, never a lifetime. There is no OOD classification and no empirical
risk threshold in serving (a threshold derived from post-hoc official-test
error analysis is NOT used to drive prospective serving behavior):

- ``history_is_padded``: objective — observed cycles < model window, the window
  is left-padded in the shared representation;
- ``n_padded_timesteps``: max(model_window - observed_cycles, 0).

The conformal interval on arbitrary uploaded trajectories is an ENGINEERING
EXTRAPOLATION. The calibration engines were held out from V2.2 fitting and
model selection, but were inspected during earlier project iterations, so the
interval is an empirically calibrated uncertainty interval rather than a
pristine one-shot external conformal guarantee.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rul_prediction.benchmark.v2 import ROOT, make_predictor

ALPHA = 0.1
CALIBRATION_METHOD = ("Engine-cluster split-conformal: one maximum-error score per "
                      "held-out calibration engine across five predefined lifecycle "
                      "checkpoints (0.25/0.45/0.65/0.80/0.95), 15 engines. Calibration "
                      "engines were inspected during earlier project iterations, so the "
                      "interval is empirically calibrated, not a pristine one-shot "
                      "external guarantee.")
UNCERTAINTY_DISCLOSURE = ("Prediction interval calibrated on held-out engines at five "
                          "predefined lifecycle checkpoints (empirical V2.2 calibration, "
                          "not a pristine one-shot guarantee). Use on arbitrary uploaded "
                          "trajectories is an engineering extrapolation.")


def load_deployment_q(alpha: float = ALPHA) -> float:
    """Final serving ``q`` from the TRACKED deployment config.

    Cross-checks against the experiment quantiles CSV when it exists locally
    (the CSV is the audit source; the config is the serving source).
    """
    cfg = yaml.safe_load((ROOT / "configs" / "deployment_v2_2_fd001.yaml")
                         .read_text(encoding="utf-8"))
    assert cfg["methodology_version"] == "2.2"
    u = cfg["uncertainty"]
    assert u["method"] == "engine_cluster_conformal"
    q = float(u["q_by_alpha"][str(alpha)])
    csv_path = ROOT / "experiments" / "v2_2" / "fd001_conformal_quantiles.csv"
    if csv_path.exists():
        table = pd.read_csv(csv_path)
        csv_q = float(table.set_index("alpha").loc[float(alpha), "q"])
        assert abs(q - csv_q) < 1e-4, (
            f"deployment config q={q} disagrees with audit CSV q={csv_q}")
    return q


class V2Predictor:
    """One-terminal-prediction-per-engine serving wrapper (V2.2)."""

    def __init__(self, alpha: float = ALPHA, q_cycles: float | None = None) -> None:
        from joblib import load as load_joblib
        from tensorflow import keras

        cfg = yaml.safe_load((ROOT / "configs" / "final_model_v2_2_fd001.yaml")
                             .read_text(encoding="utf-8"))
        assert cfg["methodology_version"] == "2.2"
        self.candidate = cfg["model"]["candidate_name"]
        self.model_version = f"v2.2-{self.candidate}"
        self.arch = self.candidate.split("_")[0]
        self._model_name = "xgboost" if self.arch == "xgb" else self.arch
        self.window = int(cfg["model"]["window"])
        model_file = ROOT / "models" / "v2_2" / f"fd001_{self.candidate}.keras"
        if self._model_name in ("rf", "xgboost"):
            model_file = ROOT / "models" / "v2_2" / f"fd001_{self.candidate}.joblib"
            self.model = load_joblib(model_file)
        else:
            self.model = keras.models.load_model(model_file)
        self.scaler = load_joblib(ROOT / "models" / "v2_2" / "fd001_scaler.joblib")
        self._predict_one = make_predictor(self._model_name, self.model, self.scaler,
                                           self.window)
        self.q_cycles = load_deployment_q(alpha) if q_cycles is None else float(q_cycles)
        self.alpha = alpha
        self.calibration_method = CALIBRATION_METHOD
        self.uncertainty_disclosure = UNCERTAINTY_DISCLOSURE

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-engine terminal prediction on a C-MAPSS frame.

        Returns one row per engine with the raw-RUL prediction, the 90%
        engine-cluster conformal interval, the objective padding/history
        fields, the model version and the calibration method. No OOD
        classification; no empirical risk flag.
        """
        if not {"engine_id", "cycle"}.issubset(frame.columns):
            raise ValueError("frame must contain 'engine_id' and 'cycle' columns")
        rows = []
        for engine, g in frame.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
            history = g.reset_index(drop=True)
            cutoff = int(history["cycle"].iloc[-1])
            pred = float(self._predict_one(history, cutoff))
            n = int(len(history))
            n_padded = max(0, self.window - n)
            rows.append({
                "engine_id": int(engine),
                "model_version": self.model_version,
                "n_cycles_observed": n,
                "history_is_padded": bool(n < self.window),
                "n_padded_timesteps": n_padded,
                "prediction_raw_rul": round(pred, 2),
                "lo_90": round(pred - self.q_cycles, 2),
                "hi_90": round(pred + self.q_cycles, 2),
                "interval_width_90": round(2 * self.q_cycles, 2),
                "calibration_method": self.calibration_method,
            })
        return pd.DataFrame(rows)


def limited_history_warning(n_observed: int, window: int) -> str | None:
    """Exact warning text for short observed histories (None when full window)."""
    if n_observed >= window:
        return None
    padded = window - n_observed
    return (f"Limited observed history: only {n_observed} cycles observed; "
            f"window {window} -> {padded} timesteps padded.")