"""Methodology V2 primary raw-RUL benchmark (window 30, no cap).

Runs mean/linear/random-forest/XGBoost/LSTM/GRU/TCN with IDENTICAL:
    - train partition (70 V2 engines), scaler (train-only), raw-RUL target
    - fixed 75-row validation pseudo-test manifest (one terminal prediction
      per engine/cutoff; every model evaluated on the same rows)
Writes:
    experiments/v2_results.csv              (one row per model)
    experiments/v2_validation_predictions.csv (raw predictions per manifest row)
The official FD001 test set is NOT contacted here.

Usage:
    .venv/Scripts/python.exe scripts/run_v2_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.preprocessing import transform
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.splitting import SEED, read_v2_split_file
from rul_prediction.data.v2_preprocessing import build_v2_train_sequences
from rul_prediction.data.windows import build_window, window_mask
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.features.v2_features import extract_v2_features
from rul_prediction.models.baseline import MeanBaseline, linear_regressor, random_forest
from rul_prediction.models.xgboost_model import xgboost_regressor

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 30
TARGET_MODE = "raw"
MODELS = ("mean", "linear", "rf", "xgboost", "lstm", "gru", "tcn")


def _train_sequences(dataset: str, data_dir: Path, split_path: Path, window: int):
    """Scaled padded sequences + raw targets for the 70 training engines (from V2 artifacts)."""
    frame = load_train(dataset, data_dir)
    train_ids, _, _ = read_v2_split_file(split_path)
    v2_dir = ROOT / "data" / "processed" / "v2" / dataset / "raw"
    from joblib import load as load_joblib

    scaler = load_joblib(v2_dir / f"{dataset}_scaler.joblib")
    npz = np.load(v2_dir / f"{dataset}_train_sequences.npz")
    return frame, train_ids, scaler, npz


def _partition_sequences(frame, engine_ids, scaler, window):
    """Scaled padded sequences for a partition, built with the SAME shared builder."""
    rows = frame[frame["engine_id"].isin(engine_ids)]
    scaled = transform(rows, SENSOR_COLUMNS, scaler)
    from rul_prediction.data.v2_preprocessing import add_raw_rul

    rul = add_raw_rul(rows)["rul"].to_numpy(dtype=np.float32)
    return build_v2_train_sequences(scaled, rows["engine_id"].to_numpy(), rul, window)


def _classical_features(X, engine_ids, n_observed, window):
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


def _metrics(y_true, y_pred, n_samples: int) -> dict:
    total = nasa_score(y_true, y_pred)
    return {
        "RMSE": round(rmse(y_true, y_pred), 4),
        "MAE": round(mae(y_true, y_pred), 4),
        "R2": round(r2(y_true, y_pred), 4),
        "NASA_total": round(total, 2),
        "NASA_mean": round(total / n_samples, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Methodology V2 raw-RUL benchmark")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    window = args.window
    split_path = Path(args.splits_dir) / f"{args.dataset}_v2_seed{SEED}.json"
    data_dir = Path(args.data_dir)

    frame, train_ids, scaler, npz = _train_sequences(args.dataset, data_dir, split_path, window)
    X, y, train_seq_ids, n_observed, masks = npz["X"], npz["y"], npz["engine_ids"], npz["n_observed"], npz["masks"]
    train_ids, validation_ids, calibration_ids = read_v2_split_file(split_path)

    manifest = load_manifest(Path(args.splits_dir) / "fd001_v2_validation_cutoffs.csv")
    assert len(manifest) == 75
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(validation_ids)].groupby("engine_id")
    }

    F_train, feature_names = _classical_features(X, train_seq_ids, n_observed, window)
    X_val, y_val, val_seq_ids, n_val_obs, m_val = _partition_sequences(
        frame, validation_ids, scaler, window)
    F_val, _ = _classical_features(X_val, val_seq_ids, n_val_obs, window)

    def predict_classical(model):
        def predict_one(history, cutoff):
            observed = scaler.transform(history[SENSOR_COLUMNS].to_numpy(dtype=float))
            features, _ = extract_v2_features(observed, cutoff)
            return float(model.predict(features[None])[0])
        return predict_one

    def predict_sequence(model):
        def predict_one(history, cutoff):
            scaled = scaler.transform(history[SENSOR_COLUMNS].to_numpy(dtype=float))
            win, n_obs, _ = build_window(scaled, cutoff, window)
            return float(model.predict([win[None], window_mask(n_obs, window)[None]], verbose=0)[0, 0])
        return predict_one

    results = []
    predictions_rows = []
    out_csv = ROOT / "experiments" / "v2_results.csv"
    done_ids = set()
    if out_csv.exists():
        import csv as _csv

        done_ids = {r["experiment_id"] for r in _csv.DictReader(out_csv.open(encoding="utf-8"))}
        print(f"resuming: {len(done_ids)} experiment(s) already in {out_csv}")

    for model_name in args.models:
        experiment_id = f"v2_{model_name}_w{window}_s{SEED}"
        if experiment_id in done_ids:
            print(f"skipping {model_name} (already present as {experiment_id})")
            continue
        start = time.perf_counter()
        parameters, notes, feature_count = "", "", 0

        if model_name == "mean":
            model = MeanBaseline().fit(F_train, y)
            predictor = lambda history, cutoff: model.value
            feature_count, parameters, notes = 0, "none", "constant prediction = mean of train raw-RUL targets"
        elif model_name in ("linear", "rf", "xgboost"):
            if model_name == "linear":
                model = linear_regressor(SEED).fit(F_train, y)
                parameters = "ols"
            elif model_name == "rf":
                model = random_forest(SEED).fit(F_train, y)
                parameters = "n_estimators=300"
            else:
                model = xgboost_regressor(SEED)
                model.fit(F_train, y, eval_set=[(F_val, y_val)], verbose=False)
                parameters = "n_estimators=500,depth=6,lr=0.05,es=30"
            predictor = predict_classical(model)
            feature_count = len(feature_names)
            notes = "variable-history features (observed cycles up to cutoff only)"
        else:
            from rul_prediction.models.v2_models import v2_gru, v2_lstm, v2_tcn
            from rul_prediction.training.trainer import set_seed
            from tensorflow import keras

            set_seed(SEED)
            builder = {"lstm": v2_lstm, "gru": v2_gru, "tcn": v2_tcn}[model_name]
            model = builder(window, X.shape[2], seed=SEED)
            callbacks = [keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True)]
            model.fit([X, masks], y, validation_data=([X_val, m_val], y_val),
                      batch_size=args.batch_size, epochs=args.epochs,
                      callbacks=callbacks, verbose=1)
            predictor = predict_sequence(model)
            feature_count = int(X.shape[2])
            parameters = f"layers={builder.__name__};bs={args.batch_size};es_patience={args.patience}"
            notes = "padded+masked training sequences (shared history builder); raw-RUL targets"

        pred = evaluate_manifest(manifest, trajectories, predictor)
        assert pred.shape == (len(manifest),)
        training_time = round(time.perf_counter() - start, 2)
        metrics = _metrics(manifest["true_raw_rul"].to_numpy(), pred, len(manifest))

        results.append({
            "experiment_id": experiment_id,
            "model": model_name,
            "target_mode": TARGET_MODE,
            "window": window,
            "seed": SEED,
            "train_engine_count": len(train_ids),
            "validation_engine_count": len(validation_ids),
            "validation_sample_count": len(manifest),
            **metrics,
            "training_time": training_time,
            "feature_count": feature_count,
            "parameters": parameters,
            "notes": notes,
        })
        for row, p in zip(manifest.itertuples(index=False), pred):
            predictions_rows.append({
                "model": model_name, "engine_id": int(row.engine_id),
                "cutoff_cycle": int(row.cutoff_cycle),
                "true_raw_rul": float(row.true_raw_rul), "prediction": float(p),
            })
        print(f"{model_name:8s} RMSE={metrics['RMSE']:8.4f} MAE={metrics['MAE']:8.4f} "
              f"R2={metrics['R2']:8.4f} NASA_total={metrics['NASA_total']:9.2f} "
              f"NASA_mean={metrics['NASA_mean']:8.4f} time={training_time}s")

    if results:
        with out_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            if not done_ids:
                writer.writeheader()
            writer.writerows(results)
        pred_csv = ROOT / "experiments" / "v2_validation_predictions.csv"
        pred_existing = set()
        if pred_csv.exists():
            pred_existing = {
                (r["model"], int(r["engine_id"]), int(r["cutoff_cycle"]))
                for r in csv.DictReader(pred_csv.open(encoding="utf-8"))
            }
        with pred_csv.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(predictions_rows[0].keys()))
            if not pred_existing:
                writer.writeheader()
            for row in predictions_rows:
                key = (row["model"], row["engine_id"], row["cutoff_cycle"])
                if key not in pred_existing:
                    writer.writerow(row)
        print(f"\nwrote -> {out_csv}")
        print(f"wrote -> {pred_csv}")
    print("official FD001 test set NOT contacted")


if __name__ == "__main__":
    main()