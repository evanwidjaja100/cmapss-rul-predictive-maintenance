"""Phase V2-8: split-conformal calibration of the frozen model.

Calibration set = the 75 fixed calibration-manifest rows (15 engines x 5
lifecycle fractions), untouched by every prior phase. Nonconformity scores
are the frozen model's absolute residuals on those rows; the ICP quantile q
(conformal_quantile) gives intervals y_hat +/- q with marginal coverage
>= 1 - alpha on exchangeable data.

Evaluates, for alpha in (0.1, 0.2, 0.3):
  * coverage + interval width on calibration (by construction), on the 75
    validation-manifest rows, and on the 100-engine official test (POST-HOC;
    exchangeability is violated there -> coverage expected below nominal)
  * the EARLY-WARNING value y_hat - q as a prediction: RMSE/MAE/NASA vs raw
    RUL on validation and official test (the conservative-bias tradeoff
    identified in Phase V2-6)

Writes:
    reports/tables/v2_conformal_calibration.csv  (tracked)
    experiments/v2_calibration_scores.csv        (gitignored)
    experiments/v2_conformal_intervals.csv       (gitignored, alpha=0.1)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from keras import models as keras_models

from rul_prediction.benchmark.v2 import (ROOT, evaluate_official_test,
                                         load_v2_artifacts, make_predictor)
from rul_prediction.evaluation.conformal import conformal_quantile, interval_coverage
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

WINDOW = 45
MODEL_NAME = "gru"
MODEL_PATH = ROOT / "models" / "v2_frozen_gru_w45_huber.keras"
ALPHAS = (0.1, 0.2, 0.3)
RESULT_CSV = ROOT / "reports" / "tables" / "v2_conformal_calibration.csv"
SCORES_CSV = ROOT / "experiments" / "v2_calibration_scores.csv"
INTERVALS_CSV = ROOT / "experiments" / "v2_conformal_intervals.csv"


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    total = nasa_score(y_true, y_pred)
    return {"RMSE": round(rmse(y_true, y_pred), 4),
            "MAE": round(mae(y_true, y_pred), 4),
            "NASA_total": round(total, 2),
            "NASA_mean": round(total / len(y_true), 4)}


def main() -> None:
    a = load_v2_artifacts()
    frame, scaler = a["frame"], a["scaler"]
    calibration_ids = a["calibration_ids"]
    manifest, trajectories = a["manifest"], a["trajectories"]
    model = keras_models.load_model(MODEL_PATH)
    predictor = make_predictor(MODEL_NAME, model, scaler, WINDOW)

    cal_path = ROOT / "experiments" / "splits" / "fd001_v2_calibration_cutoffs.csv"
    cal_manifest = pd.read_csv(cal_path, dtype={"engine_id": int, "cutoff_cycle": int})
    cal_trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(calibration_ids)].groupby("engine_id")
    }
    assert len(cal_manifest) == 75 and len(cal_trajectories) == len(calibration_ids)

    cal_pred = evaluate_manifest(cal_manifest, cal_trajectories, predictor)
    cal_true = cal_manifest["true_raw_rul"].to_numpy()
    scores = np.abs(cal_pred - cal_true)

    val_true = manifest["true_raw_rul"].to_numpy()
    val_pred = evaluate_manifest(manifest, trajectories, predictor)
    official_metrics, official_rows = evaluate_official_test(MODEL_NAME, model, scaler, WINDOW)
    off_true = np.array([r["true_rul_official"] for r in official_rows], float)
    off_pred = np.array([r["prediction"] for r in official_rows], float)

    print(f"calibration residuals: n={len(scores)}  mean {scores.mean():.2f}  "
          f"median {np.median(scores):.2f}  max {scores.max():.2f}")
    plain_val = metrics(val_true, val_pred)
    plain_off = metrics(off_true, off_pred)
    print(f"plain predictions  validation {plain_val}  official {plain_off}")

    rows = []
    for alpha in ALPHAS:
        q = conformal_quantile(scores, alpha)
        row = {
            "alpha": alpha,
            "target_coverage": round(1 - alpha, 2),
            "q_cycles": round(q, 4),
            "mean_interval_width": round(2 * q, 4),
            "calibration_coverage": round(interval_coverage(cal_true, cal_pred, q), 4),
            "validation_coverage": round(interval_coverage(val_true, val_pred, q), 4),
            "official_coverage": round(interval_coverage(off_true, off_pred, q), 4),
        }
        for name, t, p in (("validation", val_true, val_pred),
                           ("official", off_true, off_pred)):
            m = metrics(t, p - q)  # lower bound as the early-warning prediction
            row[f"{name}_lb_RMSE"] = m["RMSE"]
            row[f"{name}_lb_MAE"] = m["MAE"]
            row[f"{name}_lb_NASA_total"] = m["NASA_total"]
        row["validation_plain_RMSE"] = plain_val["RMSE"]
        row["validation_plain_NASA_total"] = plain_val["NASA_total"]
        row["official_plain_RMSE"] = plain_off["RMSE"]
        row["official_plain_NASA_total"] = plain_off["NASA_total"]
        rows.append(row)
        print(f"alpha={alpha}: q={q:.2f}  cal-cov {row['calibration_coverage']:.3f}  "
              f"val-cov {row['validation_coverage']:.3f}  off-cov {row['official_coverage']:.3f}")
        print(f"  lb-as-prediction: validation {metrics(val_true, val_pred - q)}")
        print(f"  lb-as-prediction: official   {metrics(off_true, off_pred - q)}")

    assert rows[0]["calibration_coverage"] >= 1 - ALPHAS[0] - 1e-9, "calibration coverage must meet guarantee"

    with RESULT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote -> {RESULT_CSV}")

    with SCORES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["engine_id", "cutoff_cycle", "true_raw_rul",
                                                "prediction", "abs_residual"])
        writer.writeheader()
        for r, t, p in zip(cal_manifest.itertuples(index=False), cal_true, cal_pred):
            writer.writerow({"engine_id": int(r.engine_id), "cutoff_cycle": int(r.cutoff_cycle),
                             "true_raw_rul": float(t), "prediction": float(p),
                             "abs_residual": float(abs(p - t))})
    print(f"wrote -> {SCORES_CSV}")

    q10 = conformal_quantile(scores, 0.1)
    interval_rows = []
    for name, t, p, ids in (("validation", val_true, val_pred, manifest["engine_id"].to_numpy()),
                            ("official", off_true, off_pred,
                             np.array([r["engine_id"] for r in official_rows]))):
        for e, y, ph in zip(ids, t, p):
            interval_rows.append({"sample": name, "engine_id": int(e), "true_rul": float(y),
                                  "prediction": float(ph), "lo_90": float(ph - q10),
                                  "hi_90": float(ph + q10)})
    with INTERVALS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(interval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(interval_rows)
    print(f"wrote -> {INTERVALS_CSV}  (alpha=0.1)")


if __name__ == "__main__":
    main()