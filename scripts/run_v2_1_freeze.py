"""Methodology V2.1: freeze the CV-selected FD001 model and evaluate post-hoc.

Reads configs/final_model_v2_1_fd001.yaml (source of truth for every training
parameter and the CV-derived expectations), retrains the GRU w45 huber model
on the 85 development engines ONLY (scaler fit on those 85 rows only, the 15
calibration engines never touch training), saves model + scaler, and reports
post-hoc official-test metrics.

Artifacts:
    models/v2_1/fd001_gru_w45_huber.keras
    models/v2_1/fd001_scaler.joblib
    reports/tables/v2_1_fd001_official.csv
    experiments/v2_1/fd001_final_model_metrics.json

Integrity checks: config's CV metrics must match experiments/v2_1/fd001_cv_summary.csv,
and the engine-ID hashes must match the CV manifest payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump as dump_joblib
from sklearn.preprocessing import StandardScaler

from rul_prediction.benchmark.v2 import (
    ROOT,
    evaluate_official_test,
    make_predictor,
    partition_sequences,
    train_model,
)
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.preprocessing import transform
from rul_prediction.data.v2_1_splits import read_v2_1_cv_manifest
from rul_prediction.data.v2_preprocessing import add_raw_rul, build_v2_train_sequences
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze V2.1 FD001 final model")
    parser.add_argument("--config", default="configs/final_model_v2_1_fd001.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    assert cfg["methodology"] == "v2.1" and cfg["dataset"] == "FD001"
    model_cfg = cfg["model"]

    cv_summary = pd.read_csv("experiments/v2_1/fd001_cv_summary.csv")
    selected = cv_summary[cv_summary.candidate_id == "gru_w45_huber"].iloc[0]
    expected = cfg["selection"]["cv_metrics"]
    assert abs(selected.RMSE_mean - expected["RMSE_mean"]) < 1e-4, "config CV metrics stale"
    assert abs(selected.NASA_total_mean - expected["NASA_total_mean"]) < 1e-4

    payload = read_v2_1_cv_manifest(Path(args.splits_dir) / "fd001_v2_1_group_cv_seed42.json")
    assert payload["development_sha256"] == cfg["splits"]["development_engine_ids_sha256"]
    assert payload["calibration_sha256"] == cfg["splits"]["calibration_engine_ids_sha256"]
    dev_ids = set(payload["development_engine_ids"])
    cal_ids = set(payload["calibration_engine_ids"])

    frame = load_train("FD001", args.data_dir)
    assert set(frame["engine_id"].unique()) == dev_ids | cal_ids

    window, seed = model_cfg["window"], model_cfg["seed"]
    scaler = StandardScaler().fit(
        frame[frame["engine_id"].isin(dev_ids)][SENSOR_COLUMNS].to_numpy(dtype=float))

    rows = frame[frame["engine_id"].isin(dev_ids)]
    X, y, ids, n_observed, masks = build_v2_train_sequences(
        transform(rows, SENSOR_COLUMNS, scaler),
        rows["engine_id"].to_numpy(), add_raw_rul(rows)["rul"].to_numpy(dtype=np.float32),
        window)

    cal_rows = frame[frame["engine_id"].isin(cal_ids)]
    X_val, y_val, ids_val, n_val_obs, m_val = partition_sequences(frame, cal_ids, scaler, window)

    start = time.perf_counter()
    model, parameters, notes, feature_count = train_model(
        "gru", window, X.shape[2], {"loss": model_cfg["loss"]}, X, y, masks,
        None, None, y_val, X_val=X_val, m_val=m_val, seed=seed,
        epochs=model_cfg["epochs"], batch_size=model_cfg["batch_size"],
        patience=model_cfg["patience"])
    training_time = round(time.perf_counter() - start, 2)

    out_dir = ROOT / "models" / "v2_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "fd001_gru_w45_huber.keras")
    dump_joblib(scaler, out_dir / "fd001_scaler.joblib")
    print(f"saved model + scaler to {out_dir} ({training_time}s)")

    metrics, prediction_rows = evaluate_official_test("gru", model, scaler, window,
                                                      dataset="FD001", data_dir=args.data_dir)
    metrics.update({
        "methodology": "v2.1",
        "model": "gru_w45_huber",
        "training_engine_count": len(dev_ids),
        "training_time": training_time,
        "parameters": parameters,
        "notes": notes,
        "official_status": "post-hoc (labels inspected in V2-0 audit; never blind)",
        "cv_RMSE_mean_std": f"{expected['RMSE_mean']} +- {expected['RMSE_std']}",
        "cv_NASA_total_mean_std": f"{expected['NASA_total_mean']} +- {expected['NASA_total_std']}",
    })
    pd.DataFrame([metrics]).to_csv("reports/tables/v2_1_fd001_official.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv("reports/tables/v2_1_fd001_predictions.csv", index=False)
    Path("experiments/v2_1/fd001_final_model_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()