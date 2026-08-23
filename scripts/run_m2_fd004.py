"""Methodology M2: FD004 condition-aware experiments A/B/C/D (issue R13).

Same split (175 dev / 37 val / 37 cal), target (raw RUL), manifest
(fd004_m2_validation_cutoffs.csv) and training recipe (GRU w45 huber,
epochs 60, batch 256, patience 8, seed 42) for all variants:

    A  global scaler, 21 sensor inputs            (M1-11 baseline reproduction)
    B  global scaler on sensors + settings, 24 inputs
    C  per-regime scalers (KMeans k=6), 21 inputs
    D  C + settings + one-hot regime, 30 inputs

Success (M2_REPAIR_PLAN.md R13): prediction variance no longer collapses,
RMSE decreases and R2 improves vs A.

Results append idempotently to experiments/m2/fd004_variant_results.csv and
experiments/m2/fd004_variant_predictions.csv.

Usage: python scripts/run_m2_fd004.py [--variants A B C D]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from rul_prediction.data.condition import (
    SETTING_COLUMNS,
    condition_feature_matrix,
    fit_condition_models,
)
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.m1_preprocessing import add_raw_rul, build_m1_train_sequences
from rul_prediction.data.windows import build_window, window_mask
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.models.m1_models import m1_gru
from rul_prediction.training.trainer import set_seed

WINDOW = 45
OUT_DIR = Path("experiments/m2")

VARIANTS = {"A", "B", "C", "D"}


def build_matrix(variant: str, rows: pd.DataFrame, kmeans, cluster_scalers,
                 settings_scaler, global_scaler) -> np.ndarray:
    if variant in ("A", "B"):
        cols = SENSOR_COLUMNS + (SETTING_COLUMNS if variant == "B" else [])
        return global_scaler.transform(rows[cols].to_numpy(dtype=float)).astype(np.float32)
    matrix, _, _ = condition_feature_matrix(
        rows, kmeans, cluster_scalers, settings_scaler,
        with_settings=variant == "D", with_regime=variant == "D")
    return matrix


def make_predictor(variant: str, model, kmeans, cluster_scalers, settings_scaler,
                   global_scaler):
    def predict_one(history, cutoff):
        rows = history.sort_values("cycle").reset_index(drop=True)
        features = build_matrix(variant, rows, kmeans, cluster_scalers, settings_scaler,
                                global_scaler)
        win, n_obs, _ = build_window(features, len(rows), WINDOW)
        return float(model.predict([win[None], window_mask(n_obs, WINDOW)[None]],
                                   verbose=0)[0, 0])
    return predict_one


def run_variant(variant: str, frame: pd.DataFrame, train_ids, val_ids,
                splits_dir: str) -> tuple[dict, list[dict]]:
    print(f"== variant {variant} ==")
    manifest = load_manifest(Path(splits_dir) / "fd004_m2_validation_cutoffs.csv")
    assert len(manifest) == len(val_ids) * 5
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(val_ids)].groupby("engine_id")
    }

    if variant in ("A", "B"):
        cols = SENSOR_COLUMNS + (SETTING_COLUMNS if variant == "B" else [])
        global_scaler = StandardScaler().fit(
            frame[frame["engine_id"].isin(train_ids)][cols].to_numpy(dtype=float))
        kmeans = cluster_scalers = settings_scaler = None
    else:
        kmeans, cluster_scalers, settings_scaler = fit_condition_models(
            frame, train_ids, k=6, seed=42)
        global_scaler = None

    train_rows = frame[frame["engine_id"].isin(train_ids)].sort_values(["engine_id", "cycle"])
    X = build_matrix(variant, train_rows, kmeans, cluster_scalers, settings_scaler,
                     global_scaler)
    rul = add_raw_rul(train_rows)["rul"].to_numpy(dtype=np.float32)
    X_seq, y_seq, _, _, masks = build_m1_train_sequences(
        X, train_rows["engine_id"].to_numpy(), rul, WINDOW)

    val_rows = frame[frame["engine_id"].isin(val_ids)].sort_values(["engine_id", "cycle"])
    X_val = build_matrix(variant, val_rows, kmeans, cluster_scalers, settings_scaler,
                         global_scaler)
    y_val = add_raw_rul(val_rows)["rul"].to_numpy(dtype=np.float32)
    X_val_seq, y_val_seq, _, _, m_val = build_m1_train_sequences(
        X_val, val_rows["engine_id"].to_numpy(), y_val, WINDOW)

    start = time.perf_counter()
    set_seed(42)
    model = m1_gru(WINDOW, X.shape[1], loss="huber", seed=42)
    model.fit([X_seq, masks], y_seq, validation_data=([X_val_seq, m_val], y_val_seq),
              batch_size=256, epochs=60, callbacks=[
                  keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                                restore_best_weights=True)], verbose=0)
    training_time = round(time.perf_counter() - start, 2)

    predictor = make_predictor(variant, model, kmeans, cluster_scalers, settings_scaler,
                               global_scaler)
    pred = evaluate_manifest(manifest, trajectories, predictor)
    y_true = manifest["true_raw_rul"].to_numpy()
    row = {
        "variant": variant,
        "inputs": X.shape[1],
        "train_engine_count": len(train_ids),
        "validation_engine_count": len(val_ids),
        "validation_sample_count": len(manifest),
        "RMSE": round(float(rmse(y_true, pred)), 4),
        "MAE": round(float(mae(y_true, pred)), 4),
        "R2": round(float(r2(y_true, pred)), 4),
        "NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "prediction_std": round(float(np.std(pred)), 4),
        "training_time": training_time,
    }
    prediction_rows = [
        {"variant": variant, "engine_id": int(r.engine_id), "cutoff_cycle": int(r.cutoff_cycle),
         "true_raw_rul": float(r.true_raw_rul), "prediction": float(p)}
        for r, p in zip(manifest.itertuples(index=False), pred)
    ]
    print(f"variant {variant}: RMSE={row['RMSE']} R2={row['R2']} NASA={row['NASA_total']} "
          f"pred_std={row['prediction_std']}")
    return row, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="FD004 condition-aware experiments")
    parser.add_argument("--variants", nargs="*", default=["A", "B", "C", "D"])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    args = parser.parse_args()

    frame = load_train("FD004", args.data_dir)
    split = json.loads((Path(args.splits_dir) / "fd004_m1_seed42.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_engine_ids"])
    val_ids = set(split["validation_engine_ids"])
    assert len(train_ids) == 175 and len(val_ids) == 37

    for variant in args.variants:
        if variant not in VARIANTS:
            sys.exit(f"unknown variant {variant}")
        done = set()
        path = OUT_DIR / "fd004_variant_results.csv"
        if path.exists():
            done = set(pd.read_csv(path)["variant"])
        if variant in done:
            print(f"skip variant {variant} (already done)")
            continue
        row, pred_rows = run_variant(variant, frame, train_ids, val_ids, args.splits_dir)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        pred_path = OUT_DIR / "fd004_variant_predictions.csv"
        pd.DataFrame(pred_rows).to_csv(pred_path, mode="a",
                                       header=not pred_path.exists(), index=False)

    results = pd.read_csv(OUT_DIR / "fd004_variant_results.csv")
    print("\nFD004 variant results:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()