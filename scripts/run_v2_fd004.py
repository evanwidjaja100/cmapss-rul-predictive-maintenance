"""Phase V2-11: FD004 generalization — replicate the V2 methodology on FD004.

Reuses the exact V2 recipe from the FD001 freeze (GRU, window 45, huber,
batch 256, patience 8, raw-RUL target, seed 42) on a fresh FD004 pipeline:
new 70/15/15 engine split (seed 42), scaler fit on FD004 train engines only,
185-row validation pseudo-test manifest, official FD004 test evaluated
post-hoc.

Sealed-labels gate: RUL_FD004.txt is hashed on every run and compared to the
baseline recorded at download time (V2-11, 2026-08-15). The repository
contains no code path that reads RUL_FD004.txt before this script; the
evaluation below is therefore the FIRST read of the FD004 labels.

Writes:
    models/v2_fd004_gru_w45_huber.keras          (gitignored)
    reports/tables/v2_fd004_official.csv         (tracked)
    experiments/v2_fd004_official_predictions.csv (gitignored)
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

from rul_prediction.benchmark.v2 import (ROOT, evaluate_official_test,
                                         load_v2_artifacts, make_predictor,
                                         partition_sequences, train_model)
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

DATASET = "FD004"
MODEL_NAME = "gru"
WINDOW = 45
OVERRIDES = {"loss": "huber"}

SEALED_RUL_SHA256 = "196B836B85A95AC7FDBBF29C5FDF1657382EAFA445644D114FFAAF50DC2975E1"

MODEL_PATH = ROOT / "models" / "v2_fd004_gru_w45_huber.keras"
RESULT_CSV = ROOT / "reports" / "tables" / "v2_fd004_official.csv"
PRED_CSV = ROOT / "experiments" / "v2_fd004_official_predictions.csv"


def check_sealed_rul() -> None:
    """Label integrity pin (historical). The 'sealed/first-ever' status ended at
    V2-11 when the labels were inspected; official FD004 is now post-hoc by
    policy (V2_1_REPAIR_PLAN.md R14). The hash pin remains as an integrity check.
    """
    rul_path = ROOT / "data" / "raw" / "RUL_FD004.txt"
    digest = hashlib.sha256(rul_path.read_bytes()).hexdigest().upper()
    assert digest == SEALED_RUL_SHA256, (
        f"RUL_FD004.txt hash changed since download: {digest} vs {SEALED_RUL_SHA256} "
        f"(label integrity pin). Aborting.")
    print(f"label integrity pin passed: RUL_FD004.txt sha256 == {digest}")


def main() -> None:
    check_sealed_rul()

    artifacts = load_v2_artifacts(DATASET)
    frame, scaler = artifacts["frame"], artifacts["scaler"]
    train_ids, validation_ids = artifacts["train_ids"], artifacts["validation_ids"]
    manifest, trajectories = artifacts["manifest"], artifacts["trajectories"]
    assert len(train_ids) == 175 and len(validation_ids) == 37
    assert len(manifest) == 185

    X, y, ids, n_obs, masks = partition_sequences(frame, set(train_ids), scaler, WINDOW)
    X_val, y_val, ids_val, n_val_obs, m_val = partition_sequences(
        frame, set(validation_ids), scaler, WINDOW)

    start = time.perf_counter()
    model, parameters, notes, feature_count = train_model(
        MODEL_NAME, WINDOW, X.shape[2], OVERRIDES, X, y, masks,
        None, None, y_val, X_val=X_val, m_val=m_val)
    training_time = round(time.perf_counter() - start, 2)
    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}  ({training_time}s)")

    pred_val = evaluate_manifest(manifest, trajectories,
                                 make_predictor(MODEL_NAME, model, scaler, WINDOW))
    y_true_val = manifest["true_raw_rul"].to_numpy()
    val_metrics = {
        "validation_RMSE": round(rmse(y_true_val, pred_val), 4),
        "validation_MAE": round(mae(y_true_val, pred_val), 4),
        "validation_R2": round(r2(y_true_val, pred_val), 4),
        "validation_NASA_total": round(nasa_score(y_true_val, pred_val), 2),
    }
    print("validation manifest (185 rows):", val_metrics)

    official, prediction_rows = evaluate_official_test(MODEL_NAME, model, scaler, WINDOW,
                                                       dataset=DATASET)
    print("official FD004 test (first-ever label read, post-hoc):", official)

    row = {
        "experiment_id": "v2_fd004_gru_w45_huber",
        "model": MODEL_NAME,
        "target_mode": "raw",
        "window": WINDOW,
        "parameters": parameters,
        "training_time": training_time,
        "train_engine_count": len(train_ids),
        "validation_sample_count": len(manifest),
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


if __name__ == "__main__":
    main()