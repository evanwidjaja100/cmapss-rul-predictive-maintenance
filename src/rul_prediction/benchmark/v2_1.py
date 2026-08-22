"""Methodology V2.1 CV benchmark plumbing (engine-group 5-fold cross-validation).

See V2_1_REPAIR_PLAN.md (R5-R8). Key differences vs V2:

- development/calibration engine separation: the 15 V2 calibration engines
  stay untouched by model selection;
- 5 engine-group folds (seed 42), each fold fits its own scaler on that
  fold's training rows only (no leakage from validation engines);
- balanced lifecycle fractions 0.25/0.45/0.65/0.80/0.95 -> 85 manifest rows
  per fold (17 engines x 5 cutoffs);
- per-fold row-level metrics AND per-engine metrics (5 cutoffs per engine);
  selection uses the macro-average across folds with mean +/- std.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rul_prediction.benchmark.v2 import (
    classical_features,
    make_predictor,
    partition_sequences,
    train_model,
)
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.preprocessing import transform
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.splitting import SEED
from rul_prediction.data.v2_1_splits import (
    fold_training_engines,
    fold_validation_engines,
    read_v2_1_cv_manifest,
)
from rul_prediction.data.v2_preprocessing import add_raw_rul, build_v2_train_sequences
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

CV_CANDIDATES = [
    {"id": "gru_w45_huber", "model": "gru", "window": 45, "overrides": {"loss": "huber"}},
    {"id": "gru_w60_huber", "model": "gru", "window": 60, "overrides": {"loss": "huber"}},
    {"id": "lstm_w45_huber", "model": "lstm", "window": 45, "overrides": {"loss": "huber"}},
    {"id": "lstm_w60_huber", "model": "lstm", "window": 60, "overrides": {"loss": "huber"}},
    {"id": "rf_w60", "model": "rf", "window": 60, "overrides": None},
    {"id": "rf_w90", "model": "rf", "window": 90, "overrides": None},
    {"id": "xgb_w60_d6", "model": "xgboost", "window": 60, "overrides": {"max_depth": 6}},
    {"id": "xgb_w90_d6", "model": "xgboost", "window": 90, "overrides": {"max_depth": 6}},
]


def load_v2_1_cv_artifacts(dataset: str = "FD001", data_dir: str | Path = "data/raw",
                           splits_dir: str | Path = "experiments/splits") -> dict:
    """Frame + V2.1 CV manifest payload (validation + training ids per fold)."""
    frame = load_train(dataset, data_dir)
    manifest_payload = read_v2_1_cv_manifest(
        Path(splits_dir) / f"{dataset.lower()}_v2_1_group_cv_seed{SEED}.json")
    folds = [
        {"fold": f["fold"], "training": set(f["training_engine_ids"]),
         "validation": set(f["validation_engine_ids"])}
        for f in manifest_payload["folds"]
    ]
    assert len(folds) == 5
    return {"frame": frame, "folds": folds, "payload": manifest_payload}


def fold_scaler(frame, train_ids, fold: int, seed: int = SEED) -> StandardScaler:
    """Scaler fit on the fold's training rows only (engine-group leakage guard)."""
    rows = frame[frame["engine_id"].isin(train_ids)]
    scaler = StandardScaler().fit(rows[SENSOR_COLUMNS].to_numpy(dtype=float))
    return scaler


def _fold_training_data(frame, train_ids, scaler, window):
    rows = frame[frame["engine_id"].isin(train_ids)]
    X, y, ids, n_observed, masks = build_v2_train_sequences(
        transform(rows, SENSOR_COLUMNS, scaler),
        rows["engine_id"].to_numpy(), add_raw_rul(rows)["rul"].to_numpy(dtype=np.float32),
        window)
    F_train, _ = classical_features(X, ids, n_observed, window)
    return X, y, ids, n_observed, masks, F_train


def engine_level_metrics(manifest: pd.DataFrame, pred: np.ndarray) -> list[dict]:
    """Per-engine metrics over its 5 checkpoint rows (macro-average unit = engine)."""
    out = []
    for engine, g in manifest.assign(prediction=pred).groupby("engine_id"):
        y = g["true_raw_rul"].to_numpy()
        p = g["prediction"].to_numpy()
        out.append({
            "engine_id": int(engine),
            "n_checkpoints": int(len(g)),
            "RMSE": float(rmse(y, p)),
            "MAE": float(mae(y, p)),
            "NASA_sum": float(nasa_score(y, p)),
            "signed_bias_mean": float(np.mean(p - y)),
        })
    return out


def run_cv_fold(candidate: dict, fold: dict, frame, data_dir: str | Path,
                splits_dir: str | Path, *, epochs: int = 60, batch_size: int = 256,
                patience: int = 8, seed: int = SEED) -> dict:
    """Train candidate on fold training engines, evaluate on the fold validation manifest."""
    window = candidate["window"]
    train_ids, val_ids = fold["training"], fold["validation"]
    scaler = fold_scaler(frame, train_ids, fold["fold"], seed)
    manifest = load_manifest(Path(splits_dir) /
                             f"fd001_v2_1_fold{fold['fold']}_validation_cutoffs.csv")
    assert len(manifest) == len(val_ids) * 5
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(val_ids)].groupby("engine_id")
    }

    X, y, ids, n_observed, masks, F_train = _fold_training_data(frame, train_ids, scaler, window)
    X_val, y_val, ids_val, n_val_obs, m_val = partition_sequences(frame, val_ids, scaler, window)
    F_val, _ = classical_features(X_val, ids_val, n_val_obs, window)

    start = time.perf_counter()
    model, parameters, notes, feature_count = train_model(
        candidate["model"], window, X.shape[2], candidate["overrides"], X, y, masks,
        F_train, F_val, y_val, X_val=X_val, m_val=m_val, seed=seed,
        epochs=epochs, batch_size=batch_size, patience=patience)
    training_time = round(time.perf_counter() - start, 2)

    pred = evaluate_manifest(manifest, trajectories,
                             make_predictor(candidate["model"], model, scaler,
                                            window=window))
    y_true = manifest["true_raw_rul"].to_numpy()
    row = {
        "candidate_id": candidate["id"],
        "fold": fold["fold"],
        "model": candidate["model"],
        "window": window,
        "train_engine_count": len(train_ids),
        "validation_engine_count": len(val_ids),
        "validation_sample_count": len(manifest),
        "RMSE": round(float(rmse(y_true, pred)), 4),
        "MAE": round(float(mae(y_true, pred)), 4),
        "R2": round(float(r2(y_true, pred)), 4),
        "NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "NASA_mean": round(float(nasa_score(y_true, pred)) / len(manifest), 4),
        "signed_bias_mean": round(float(np.mean(pred - y_true)), 4),
        "training_time": training_time,
        "parameters": parameters,
        "notes": notes,
        "feature_count": feature_count,
    }
    return {"row": row,
            "engine_rows": [{"candidate_id": candidate["id"], "fold": fold["fold"], **e}
                            for e in engine_level_metrics(manifest, pred)],
            "prediction_rows": [
                {"candidate_id": candidate["id"], "fold": fold["fold"],
                 "engine_id": int(r.engine_id), "cutoff_cycle": int(r.cutoff_cycle),
                 "true_raw_rul": float(r.true_raw_rul), "prediction": float(p)}
                for r, p in zip(manifest.itertuples(index=False), pred)]}


def cv_summary(fold_rows: list[dict]) -> list[dict]:
    """Aggregate per-candidate metrics across folds: mean +/- std (macro)."""
    df = pd.DataFrame(fold_rows)
    out = []
    for cid, g in df.groupby("candidate_id"):
        base = {k: g.iloc[0][k] for k in ("candidate_id", "model", "window",
                                          "parameters", "notes")}
        for metric in ("RMSE", "MAE", "R2", "NASA_total", "NASA_mean", "signed_bias_mean",
                       "training_time"):
            base[f"{metric}_mean"] = round(float(g[metric].mean()), 4)
            base[f"{metric}_std"] = round(float(g[metric].std(ddof=1)), 4)
        out.append(base)
    return out