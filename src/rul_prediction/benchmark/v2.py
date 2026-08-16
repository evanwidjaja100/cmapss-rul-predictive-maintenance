"""Shared Methodology V2 benchmark plumbing.

Used by ``scripts/run_v2_benchmark.py`` (primary w30 benchmark) and
``scripts/run_v2_ablation.py`` (window/hyperparameter ablations) so that every
model is evaluated with the identical pipeline: same train partition, same
train-only scaler, same raw-RUL target, same fixed 75-row pseudo-test manifest.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as load_joblib

from rul_prediction.data.loader import (EXPECTED_ENGINE_COUNTS, SENSOR_COLUMNS,
                                        load_rul, load_test, load_train)
from rul_prediction.data.preprocessing import transform
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.splitting import SEED, read_v2_split_file
from rul_prediction.data.v2_preprocessing import add_raw_rul, build_v2_train_sequences
from rul_prediction.data.windows import build_window, window_mask
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.features.v2_features import extract_v2_features
from rul_prediction.models.baseline import MeanBaseline, linear_regressor, random_forest
from rul_prediction.models.xgboost_model import xgboost_regressor

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_WINDOW = 30


def load_v2_artifacts(dataset: str = "FD001", data_dir: str | Path = "data/raw",
                      splits_dir: str | Path = "experiments/splits") -> dict:
    """Load raw frame, V2 split, train-only scaler, manifest and validation trajectories."""
    frame = load_train(dataset, data_dir)
    split_path = Path(splits_dir) / f"{dataset}_v2_seed{SEED}.json"
    train_ids, validation_ids, calibration_ids = read_v2_split_file(split_path)
    v2_dir = ROOT / "data" / "processed" / "v2" / dataset / "raw"
    scaler = load_joblib(v2_dir / f"{dataset}_scaler.joblib")
    manifest = load_manifest(Path(splits_dir) / f"{dataset.lower()}_v2_validation_cutoffs.csv")
    from rul_prediction.data.pseudo_test import LIFECYCLE_FRACTIONS
    assert len(manifest) == len(validation_ids) * len(LIFECYCLE_FRACTIONS)
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(validation_ids)].groupby("engine_id")
    }
    return {
        "frame": frame, "train_ids": train_ids, "validation_ids": validation_ids,
        "calibration_ids": calibration_ids, "scaler": scaler,
        "manifest": manifest, "trajectories": trajectories,
    }


def partition_sequences(frame, engine_ids: set[int], scaler, window: int):
    """Scaled padded sequences + raw targets for a partition (shared history builder)."""
    rows = frame[frame["engine_id"].isin(engine_ids)]
    scaled = transform(rows, SENSOR_COLUMNS, scaler)
    rul = add_raw_rul(rows)["rul"].to_numpy(dtype=np.float32)
    return build_v2_train_sequences(scaled, rows["engine_id"].to_numpy(), rul, window)


def classical_features(X, engine_ids, n_observed, window: int):
    """Variable-history features for every training sequence (observed rows only)."""
    feats, names = [], None
    for engine in np.unique(engine_ids):
        block = np.flatnonzero(engine_ids == engine)
        assert np.all(np.diff(block) == 1), "sequence block must be contiguous"
        for i, idx in enumerate(block):
            observed = X[idx][window - n_observed[idx]:]
            f, names = extract_v2_features(observed, cutoff_cycle=i + 1)
            feats.append(f)
    return np.stack(feats).astype(np.float32), names


def train_model(model_name: str, window: int, n_features: int, overrides: dict | None,
                X, y, masks, F_train, F_val, y_val, *, X_val=None, m_val=None,
                seed: int = SEED, epochs: int = 60, batch_size: int = 256,
                patience: int = 8):
    """Fit one model on the identical training partition; returns (model, parameters, notes, feature_count)."""
    overrides = overrides or {}
    if model_name == "mean":
        model = MeanBaseline().fit(F_train, y)
        return model, "none", "constant prediction = mean of train raw-RUL targets", 0
    if model_name == "linear":
        model = linear_regressor(seed).fit(F_train, y)
        return model, "ols", "variable-history features (observed cycles up to cutoff only)", F_train.shape[1]
    if model_name == "rf":
        model = random_forest(seed).fit(F_train, y)
        return model, "n_estimators=300", "variable-history features (observed cycles up to cutoff only)", F_train.shape[1]
    if model_name == "xgboost":
        model = xgboost_regressor(seed)
        model.set_params(n_estimators=overrides.get("n_estimators", 500),
                         max_depth=overrides.get("max_depth", 6))
        model.fit(F_train, y, eval_set=[(F_val, y_val)], verbose=False)
        parameters = (f"n_estimators={model.n_estimators},max_depth={model.max_depth},"
                      f"lr=0.05,es=30")
        return model, parameters, "variable-history features; early stopping on validation engines", F_train.shape[1]
    if model_name in ("lstm", "gru", "tcn"):
        from tensorflow import keras

        from rul_prediction.models.v2_models import v2_gru, v2_lstm, v2_tcn
        from rul_prediction.training.trainer import set_seed

        set_seed(seed)
        builder = {"lstm": v2_lstm, "gru": v2_gru, "tcn": v2_tcn}[model_name]
        loss = overrides.get("loss", "mse")
        model = builder(window, n_features, dropout=overrides.get("dropout", 0.3 if model_name != "tcn" else 0.2),
                        loss=loss, seed=seed)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)]
        assert X_val is not None and m_val is not None, "deep models require validation sequences"
        model.fit([X, masks], y, validation_data=([X_val, m_val], y_val),
                  batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
        parameters = f"{builder.__name__};loss={loss};bs={batch_size};es_patience={patience}"
        return model, parameters, "padded+masked training sequences (shared history builder); raw-RUL targets", n_features
    raise ValueError(f"unknown model: {model_name}")


def make_predictor(model_name: str, model, scaler, window: int):
    """predict_one(history, cutoff) compatible with evaluate_manifest."""
    if model_name == "mean":
        return lambda history, cutoff: model.value
    if model_name in ("linear", "rf", "xgboost"):
        def predict_classical(history, cutoff):
            observed = scaler.transform(history[SENSOR_COLUMNS].to_numpy(dtype=float))
            features, _ = extract_v2_features(observed, cutoff)
            return float(model.predict(features[None])[0])
        return predict_classical

    def predict_sequence(history, cutoff):
        scaled = scaler.transform(history[SENSOR_COLUMNS].to_numpy(dtype=float))
        win, n_obs, _ = build_window(scaled, cutoff, window)
        return float(model.predict([win[None], window_mask(n_obs, window)[None]], verbose=0)[0, 0])
    return predict_sequence


def evaluate_official_test(model_name: str, model, scaler, window: int,
                           dataset: str = "FD001", data_dir: str | Path = "data/raw"):
    """Post-hoc metrics on the official C-MAPSS test set (labels are loaded).

    NOTE: this is NOT an "exactly once" evaluation — the official RUL labels
    were inspected during the V2-0 audit. Results must be reported as
    post-hoc. Uses the same per-engine window/mask representation as training.
    """
    test = load_test(dataset, data_dir)
    rul = load_rul(dataset, data_dir)
    engines = sorted(test["engine_id"].unique())
    assert len(engines) == len(rul) == EXPECTED_ENGINE_COUNTS[dataset]["test"], (
        f"{len(engines)} test engines vs {len(rul)} labels")
    assert engines == list(range(1, len(engines) + 1)), "engine ids must be 1..N"
    manifest = pd.DataFrame({
        "engine_id": engines,
        "cutoff_cycle": [int(test[test["engine_id"] == e]["cycle"].max()) for e in engines],
    })
    trajectories = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}
    pred = evaluate_manifest(manifest, trajectories, make_predictor(model_name, model, scaler, window))
    y_true = rul.astype(float)
    assert np.isfinite(pred).all(), "non-finite official-test predictions"
    total = nasa_score(y_true, pred)
    metrics = {
        "official_test_engine_count": len(engines),
        "official_test_RMSE": round(rmse(y_true, pred), 4),
        "official_test_MAE": round(mae(y_true, pred), 4),
        "official_test_R2": round(r2(y_true, pred), 4),
        "official_test_NASA_total": round(total, 2),
        "official_test_NASA_mean": round(total / len(engines), 4),
    }
    prediction_rows = [
        {"engine_id": int(e), "true_rul_official": float(t), "prediction": float(p)}
        for e, t, p in zip(engines, rul, pred)
    ]
    return metrics, prediction_rows


def run_experiment(experiment_id: str, model_name: str, window: int, overrides: dict | None = None,
                   *, dataset: str = "FD001", data_dir: str | Path = "data/raw",
                   splits_dir: str | Path = "experiments/splits",
                   epochs: int = 60, batch_size: int = 256, patience: int = 8,
                   seed: int = SEED) -> tuple[dict, list[dict]]:
    """Train + evaluate one model on the fixed manifest. Returns (row, prediction_rows)."""
    artifacts = load_v2_artifacts(dataset, data_dir, splits_dir)
    frame, scaler, manifest, trajectories = (
        artifacts["frame"], artifacts["scaler"], artifacts["manifest"], artifacts["trajectories"])
    train_ids, validation_ids = artifacts["train_ids"], artifacts["validation_ids"]

    train_rows = frame[frame["engine_id"].isin(train_ids)]
    X, y, ids, n_observed, masks = build_v2_train_sequences(
        transform(train_rows, SENSOR_COLUMNS, scaler),
        train_rows["engine_id"].to_numpy(), add_raw_rul(train_rows)["rul"].to_numpy(dtype=np.float32),
        window)
    F_train, names = classical_features(X, ids, n_observed, window)
    X_val, y_val, ids_val, n_val_obs, m_val = partition_sequences(frame, validation_ids, scaler, window)
    F_val, _ = classical_features(X_val, ids_val, n_val_obs, window)

    start = time.perf_counter()
    model, parameters, notes, feature_count = train_model(
        model_name, window, X.shape[2], overrides, X, y, masks,
        F_train, F_val, y_val, X_val=X_val, m_val=m_val, seed=seed,
        epochs=epochs, batch_size=batch_size, patience=patience)
    training_time = round(time.perf_counter() - start, 2)

    pred = evaluate_manifest(manifest, trajectories, make_predictor(model_name, model, scaler, window))
    assert pred.shape == (len(manifest),)
    y_true = manifest["true_raw_rul"].to_numpy()
    total = nasa_score(y_true, pred)
    row = {
        "experiment_id": experiment_id,
        "model": model_name,
        "target_mode": "raw",
        "window": window,
        "seed": seed,
        "train_engine_count": len(train_ids),
        "validation_engine_count": len(validation_ids),
        "validation_sample_count": len(manifest),
        "RMSE": round(rmse(y_true, pred), 4),
        "MAE": round(mae(y_true, pred), 4),
        "R2": round(r2(y_true, pred), 4),
        "NASA_total": round(total, 2),
        "NASA_mean": round(total / len(manifest), 4),
        "training_time": training_time,
        "feature_count": feature_count,
        "parameters": parameters,
        "notes": notes,
    }
    prediction_rows = [
        {"model": model_name, "engine_id": int(r.engine_id), "cutoff_cycle": int(r.cutoff_cycle),
         "true_raw_rul": float(r.true_raw_rul), "prediction": float(p)}
        for r, p in zip(manifest.itertuples(index=False), pred)
    ]
    return row, prediction_rows


def append_results(out_csv: Path, pred_csv: Path, rows: list[dict], prediction_rows: list[dict]) -> None:
    """Idempotent append: existing experiment_ids / (model, engine, cutoff) rows are skipped."""
    import csv

    done_ids = set()
    if out_csv.exists():
        done_ids = {r["experiment_id"] for r in csv.DictReader(out_csv.open(encoding="utf-8"))}
    with out_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not done_ids:
            writer.writeheader()
        writer.writerows([r for r in rows if r["experiment_id"] not in done_ids])
    done_keys = set()
    if pred_csv.exists():
        done_keys = {(r["model"], int(r["engine_id"]), int(r["cutoff_cycle"]))
                     for r in csv.DictReader(pred_csv.open(encoding="utf-8"))}
    new_rows = [r for r in prediction_rows
                if (r["model"], r["engine_id"], r["cutoff_cycle"]) not in done_keys]
    if new_rows:
        with pred_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(new_rows[0].keys()))
            if not done_keys:
                writer.writeheader()
            writer.writerows(new_rows)