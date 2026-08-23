"""Methodology M2: freeze the FD004 condition-aware model (variant C).

Variant C = GRU w45 huber + KMeans(k=6) regime clustering with per-regime
sensor scalers (21 inputs), trained on the 175 development engines only.
Selected over D: NASA 46,165 vs 68,981 (D is RMSE-optimal, 27.22 vs 29.37,
but 1.5x worse on the deployment-critical NASA score).

Post-hoc official evaluation: FD004 labels were inspected in M1-11
(reports/m1_fd004.md); official metrics are post-hoc, never sealed.

Artifacts:
    configs/final_model_m2_fd004.yaml
    models/m2/fd004_gru_w45_huber_condC.keras
    models/m2/fd004_conditionC.joblib  (kmeans, cluster_scalers, settings_scaler)
    reports/tables/m2_fd004_official.csv
    reports/tables/m2_fd004_predictions.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump as dump_joblib
from tensorflow import keras

from rul_prediction.benchmark.m1 import ROOT
from rul_prediction.data.condition import SETTING_COLUMNS, condition_feature_matrix
from rul_prediction.data.loader import EXPECTED_ENGINE_COUNTS, load_rul, load_test, load_train
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.m1_preprocessing import add_raw_rul, build_m1_train_sequences
from rul_prediction.data.windows import build_window, window_mask
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.models.m1_models import m1_gru
from rul_prediction.training.trainer import set_seed

from run_m2_fd004 import build_matrix  # exact variant-C feature pipeline

WINDOW = 45


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze M2 FD004 condition-C model")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    args = parser.parse_args()

    frame = load_train("FD004", args.data_dir)
    split = json.loads((Path(args.splits_dir) / "fd004_m1_seed42.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_engine_ids"])
    assert len(train_ids) == 175

    from rul_prediction.data.condition import fit_condition_models
    kmeans, cluster_scalers, settings_scaler = fit_condition_models(frame, train_ids, k=6, seed=42)

    train_rows = frame[frame["engine_id"].isin(train_ids)].sort_values(["engine_id", "cycle"])
    X = build_matrix("C", train_rows, kmeans, cluster_scalers, settings_scaler, None)
    rul = add_raw_rul(train_rows)["rul"].to_numpy(dtype=np.float32)
    X_seq, y_seq, _, _, masks = build_m1_train_sequences(
        X, train_rows["engine_id"].to_numpy(), rul, WINDOW)

    val_ids = set(split["validation_engine_ids"])
    val_rows = frame[frame["engine_id"].isin(val_ids)].sort_values(["engine_id", "cycle"])
    X_val = build_matrix("C", val_rows, kmeans, cluster_scalers, settings_scaler, None)
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

    out_dir = ROOT / "models" / "m2"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "fd004_gru_w45_huber_condC.keras")
    dump_joblib({"kmeans": kmeans, "cluster_scalers": cluster_scalers,
                 "settings_scaler": settings_scaler},
                out_dir / "fd004_conditionC.joblib")
    print(f"saved {out_dir} ({training_time}s)")

    # ---- post-hoc official FD004 ----
    test = load_test("FD004", args.data_dir)
    rul = load_rul("FD004", args.data_dir).astype(float)
    engines = sorted(test["engine_id"].unique())
    assert len(engines) == len(rul) == EXPECTED_ENGINE_COUNTS["FD004"]["test"]
    test_manifest = pd.DataFrame({
        "engine_id": engines,
        "cutoff_cycle": [int(test[test["engine_id"] == e]["cycle"].max()) for e in engines]})
    test_traj = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}

    def predict_one(history, cutoff):
        rows = history.sort_values("cycle").reset_index(drop=True)
        features = build_matrix("C", rows, kmeans, cluster_scalers, settings_scaler, None)
        win, n_obs, _ = build_window(features, len(rows), WINDOW)
        return float(model.predict([win[None], window_mask(n_obs, WINDOW)[None]], verbose=0)[0, 0])

    pred = evaluate_manifest(test_manifest, test_traj, predict_one)
    total = nasa_score(rul, pred)
    metrics = {
        "methodology": "m2", "dataset": "FD004", "variant": "C",
        "model": "gru_w45_huber_condC",
        "official_status": "post-hoc (labels inspected in M1-11; never sealed)",
        "official_test_engine_count": len(engines),
        "official_test_RMSE": round(float(rmse(rul, pred)), 4),
        "official_test_MAE": round(float(mae(rul, pred)), 4),
        "official_test_R2": round(float(r2(rul, pred)), 4),
        "official_test_NASA_total": round(total, 2),
        "official_test_NASA_mean": round(total / len(engines), 4),
        "training_time": training_time,
        "variant_results": {
            "A_RMSE": 71.9264, "A_R2": -0.2293, "A_NASA": 2554269.56,
            "C_RMSE": 29.3719, "C_R2": 0.7950, "C_NASA": 46165.29,
            "D_RMSE": 27.2231, "D_R2": 0.8239, "D_NASA": 68980.96,
        },
    }
    pd.DataFrame([metrics]).to_csv("reports/tables/m2_fd004_official.csv", index=False)
    pd.DataFrame({"engine_id": engines, "true_rul_official": rul,
                  "prediction": pred}).to_csv(
        "reports/tables/m2_fd004_predictions.csv", index=False)
    Path("experiments/m2/fd004_final_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()