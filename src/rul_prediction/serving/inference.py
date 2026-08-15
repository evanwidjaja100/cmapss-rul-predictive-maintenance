"""Deployment-ready inference for the frozen Phase 9 RUL model.

Mirrors the Phase 9 evaluation pipeline (scripts/final_evaluation.py): same
scaling (train-only scaler), same 90-cycle windows with zero-padding for
short units, same 169 engineered features, same [0, cap] clipping.
tests/test_inference_golden.py keeps this path byte-identical to the
one-time official test evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from xgboost import XGBRegressor

from rul_prediction.data.loader import load_test
from rul_prediction.data.preprocessing import load_scaler, transform
from rul_prediction.features.engineered_features import extract_features

ROOT = Path(__file__).resolve().parents[3]


def blocks(engine_ids: np.ndarray):
    start = 0
    engine = engine_ids[0]
    for k in range(1, len(engine_ids)):
        if engine_ids[k] != engine:
            yield engine, k - start
            start, engine = k, engine_ids[k]
    yield engine, len(engine_ids) - start


def final_cycles(engine_ids: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(len(engine_ids), dtype=int)
    i = 0
    for engine, _ in blocks(engine_ids):
        block_len = int(np.sum(engine_ids == engine))
        out[i : i + block_len] = window + np.arange(block_len)
        i += block_len
    return out


def test_windows(scaled: np.ndarray, unit_ids: np.ndarray, window: int):
    """Per-unit sliding windows; units shorter than `window` are left-padded
    with zeros (scaled mean ~ 0) so the window ends at the last cycle."""
    Xs, n_cycles, padded = [], [], []
    for unit in np.unique(unit_ids):
        block = scaled[unit_ids == unit]
        n = len(block)
        if n < window:
            pad = np.zeros((window - n, block.shape[1]), dtype=np.float32)
            block = np.concatenate([pad, block])
            Xs.append(block[np.newaxis].astype(np.float32))
            n_cycles.append(n)
            padded.append(True)
        else:
            Xs.append(block[np.newaxis, n - window : n].astype(np.float32))
            n_cycles.append(n)
            padded.append(False)
    return np.concatenate(Xs, axis=0), np.array(n_cycles), np.array(padded)


class RulPredictor:
    """Frozen-model RUL predictor. Stateless after construction; thread-safe."""

    def __init__(self, config_path: str | Path = ROOT / "configs" / "final_model.yaml"):
        self.config_path = Path(config_path)
        config = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.dataset = config["dataset"]
        self.variant = config["variant"]
        self.window = int(config["window"])
        self.max_rul = int(config["max_rul"])
        self._validate_config(config)

        variant_dir = ROOT / "data" / "processed" / f"{self.dataset}_{self.variant}"
        model_path = ROOT / "models" / "final" / f"{self.dataset}_final_model.joblib"
        for path in (variant_dir, model_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"missing artifact {path} - run scripts/final_evaluation.py first")
        self.scaler = load_scaler(variant_dir / f"{self.dataset}_scaler.joblib")
        self.model = XGBRegressor()
        self.model.load_model(str(model_path))
        self.sensor_cols = [c for c in load_test(self.dataset).columns if c.startswith("sensor_")]

    def _validate_config(self, config: dict) -> None:
        frozen = {"model": "xgboost", "window": 90, "max_rul": 45, "variant": "w90_c45_all"}
        for key, expected in frozen.items():
            if config[key] != expected:
                raise ValueError(f"config {key}={config[key]} != frozen {expected}")

    def predict(self, frame: pd.DataFrame) -> dict:
        """Predict RUL per unit from a C-MAPSS-formatted frame.

        Frame columns: engine_id, cycle, setting_1..3, sensor_1..21
        (or plain positional columns 0..25 in the same order).
        Returns {unit_id, n_cycles, padded_short, prediction}.
        """
        expected = ["engine_id", "cycle", *[f"setting_{i}" for i in range(1, 4)],
                    *self.sensor_cols]
        if list(frame.columns) != expected:
            frame = frame.copy()
            frame.columns = expected[: len(frame.columns)]
        scaled = transform(frame, self.sensor_cols, self.scaler)
        unit_ids = frame["engine_id"].to_numpy()
        X, n_cycles, padded = test_windows(scaled, unit_ids, self.window)
        features, _ = extract_features(X, n_cycles.astype(int))
        pred = np.clip(self.model.predict(features), 0, self.max_rul)
        return {
            "unit_id": [int(u) for u in np.unique(unit_ids)],
            "n_cycles": n_cycles.tolist(),
            "padded_short": padded.tolist(),
            "prediction": np.round(pred, 4).tolist(),
        }

    def predict_file(self, input_path: str | Path) -> dict:
        frame = pd.read_csv(input_path, sep=r"\s+", header=None, engine="python")
        frame.columns = ["engine_id", "cycle", *[f"setting_{i}" for i in range(1, 4)],
                         *self.sensor_cols]
        return self.predict(frame)