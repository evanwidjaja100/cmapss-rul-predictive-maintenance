"""Phase V2-5: freeze the V2-4-selected model and evaluate post-hoc on the official FD001 test set.

Frozen configuration = the selected ablation run ``v2_gru_w45_losshuber_s42``:
GRU, window 45, huber loss, batch 256, patience 8, raw-RUL target, seed 42.
The model is retrained with the EXACT same procedure (train engines only,
early stopping on validation-engine sequences) and saved to ``models/``.

The official FD001 test set is contacted here for the first time in V2, and
its labels were already inspected during the V2-0 audit -> this evaluation is
reported as POST-HOC, never "exactly once". Calibration engines are untouched.

Writes:
    models/v2_frozen_gru_w45_huber.keras                 (gitignored)
    reports/tables/v2_fd001_official.csv                 (tracked)
    experiments/v2_fd001_official_predictions.csv        (gitignored)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from rul_prediction.benchmark.v2 import (ROOT, evaluate_official_test,
                                         load_v2_artifacts, make_predictor,
                                         partition_sequences, train_model)
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

MODEL_NAME = "gru"
WINDOW = 45
OVERRIDES = {"loss": "huber"}
EXPECTED_VAL_RMSE = 13.7406  # selected run v2_gru_w45_losshuber_s42 (fixed manifest)
MAX_VAL_RMSE_DRIFT = 1.5     # retraining has small TF nondeterminism; a larger gap = procedure drift

MODEL_PATH = ROOT / "models" / "v2_frozen_gru_w45_huber.keras"
RESULT_CSV = ROOT / "reports" / "tables" / "v2_fd001_official.csv"
PRED_CSV = ROOT / "experiments" / "v2_fd001_official_predictions.csv"


def main() -> None:
    artifacts = load_v2_artifacts()
    frame, scaler = artifacts["frame"], artifacts["scaler"]
    train_ids, validation_ids = artifacts["train_ids"], artifacts["validation_ids"]
    manifest, trajectories = artifacts["manifest"], artifacts["trajectories"]

    X, y, ids, n_obs, masks = partition_sequences(frame, set(train_ids), scaler, WINDOW)
    X_val, y_val, ids_val, n_val_obs, m_val = partition_sequences(
        frame, set(validation_ids), scaler, WINDOW)

    start = time.perf_counter()
    model, parameters, notes, feature_count = train_model(
        MODEL_NAME, WINDOW, X.shape[2], OVERRIDES, X, y, masks,
        None, None, y_val, X_val=X_val, m_val=m_val)
    training_time = round(time.perf_counter() - start, 2)
    assert MODEL_NAME in ("lstm", "gru", "tcn"), "freeze expects a keras model"
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}  ({training_time}s)")

    pred_val = evaluate_manifest(manifest, trajectories,
                                 make_predictor(MODEL_NAME, model, scaler, WINDOW))
    y_true_val = manifest["true_raw_rul"].to_numpy()
    val_total = nasa_score(y_true_val, pred_val)
    val_metrics = {
        "validation_RMSE": round(rmse(y_true_val, pred_val), 4),
        "validation_MAE": round(mae(y_true_val, pred_val), 4),
        "validation_R2": round(r2(y_true_val, pred_val), 4),
        "validation_NASA_total": round(val_total, 2),
        "validation_NASA_mean": round(val_total / len(manifest), 4),
    }
    print("validation manifest:", val_metrics)
    assert abs(val_metrics["validation_RMSE"] - EXPECTED_VAL_RMSE) < MAX_VAL_RMSE_DRIFT, (
        f"frozen retrain drifted too far from selected run: "
        f"{val_metrics['validation_RMSE']} vs {EXPECTED_VAL_RMSE}")

    official, prediction_rows = evaluate_official_test(MODEL_NAME, model, scaler, WINDOW)
    print("official FD001 test (post-hoc):", official)

    row = {
        "experiment_id": "v2_frozen_gru_w45_huber",
        "model": MODEL_NAME,
        "target_mode": "raw",
        "window": WINDOW,
        "parameters": parameters,
        "training_time": training_time,
        **val_metrics,
        **official,
    }
    RESULT_CSV.parent.mkdir(exist_ok=True)
    with RESULT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote -> {RESULT_CSV}")
    with PRED_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"wrote -> {PRED_CSV}")
    print("official FD001 test set CONTACTED (post-hoc evaluation)")


if __name__ == "__main__":
    main()