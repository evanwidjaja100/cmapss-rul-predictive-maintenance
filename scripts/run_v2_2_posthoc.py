"""Methodology V2.2: post-hoc official FD001 evaluation (STRICTLY after CV ->
selection -> YAML freeze -> conformal calibration).

The official labels are permanently POST-HOC (inspected in the V2-0 audit) and
never select or tune the V2.2 model. This script only reads the frozen model.

Falsification: headline metrics are recomputed independently from the saved
prediction CSV; stored CV summaries are re-derived from the fold CSV and
compared against the config's recorded values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import load as load_joblib
from tensorflow import keras

from rul_prediction.benchmark.v2 import ROOT, make_predictor
from rul_prediction.data.loader import load_rul, load_test
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.benchmark.v2_2 import CV_CANDIDATES, apply_selection_policy, cv_summary

OUT_DIR = Path("experiments/v2_2")


def recompute_summary(fold_csv: str | Path) -> pd.DataFrame:
    """Independently recompute candidate summaries from the fold result CSV."""
    rows = pd.read_csv(fold_csv).to_dict("records")
    return pd.DataFrame(cv_summary(rows, CV_CANDIDATES))


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc official FD001 evaluation")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/final_model_v2_2_fd001.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidate = cfg["model"]["candidate_name"]
    window = cfg["model"]["window"]
    arch = candidate.split("_")[0]
    model_name = "xgboost" if arch == "xgb" else arch

    # ---- falsification 1: CV summary recomputed from fold CSV matches config ----
    recomputed = recompute_summary(OUT_DIR / "fd001_outer_fold_results.csv")
    sel_row = recomputed[recomputed.candidate_id == candidate].iloc[0]
    recorded = cfg["selection"]["cv_metrics"]
    for key in ("RMSE_mean", "NASA_mean_per_engine_mean", "signed_bias_mean_mean"):
        assert abs(sel_row[key] - recorded[key]) < 1e-4, f"config drift on {key}"

    # ---- falsification 2: selection policy reproduces the deployed candidate ----
    decision = apply_selection_policy(recomputed.to_dict("records"))
    assert decision["deployment_selection"] == candidate, (
        f"selection policy reproduces {decision['deployment_selection']}, "
        f"not deployed {candidate}")
    print(f"selection policy reproduces deployment: {candidate}")

    # ---- falsification 3: headline metrics recomputed from stored predictions ----
    pred_df = pd.read_csv(OUT_DIR / "fd001_outer_predictions.csv")
    stored_summary = pd.read_csv(OUT_DIR / "fd001_cv_summary.csv")
    for cid, g in pred_df.groupby("candidate_id"):
        y, p = g["true_raw_rul"].to_numpy(float), g["prediction"].to_numpy(float)
        re_row = recomputed[recomputed.candidate_id == cid].iloc[0]
        store_row = stored_summary[stored_summary.candidate_id == cid].iloc[0]
        assert abs(re_row["RMSE_mean"] - store_row["RMSE_mean"]) < 1e-3, cid
        assert abs(re_row["NASA_mean_per_engine_mean"] - store_row["NASA_mean_per_engine_mean"]) < 1e-3, cid

    # ---- load-time verification before deserialization (distinct errors) ----
    from rul_prediction.artifact_manifest import verify_before_load

    if model_name in ("rf", "xgboost"):
        verify_before_load(f"models/v2_2/fd001_{candidate}.joblib", root=ROOT, manifest_dataset="FD001")
        verify_before_load("models/v2_2/fd001_scaler.joblib", root=ROOT, manifest_dataset="FD001")
        model = load_joblib(ROOT / "models" / "v2_2" / f"fd001_{candidate}.joblib")
    else:
        verify_before_load(f"models/v2_2/fd001_{candidate}.keras", root=ROOT, manifest_dataset="FD001")
        verify_before_load("models/v2_2/fd001_scaler.joblib", root=ROOT, manifest_dataset="FD001")
        model = keras.models.load_model(ROOT / "models" / "v2_2" / f"fd001_{candidate}.keras")
    scaler = load_joblib(ROOT / "models" / "v2_2" / "fd001_scaler.joblib")

    test = load_test("FD001", args.data_dir)
    rul = load_rul("FD001", args.data_dir)
    engines = sorted(test["engine_id"].unique())
    test_manifest = pd.DataFrame({
        "engine_id": engines,
        "cutoff_cycle": [int(test[test["engine_id"] == e]["cycle"].max()) for e in engines]})
    test_traj = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}
    pred = evaluate_manifest(test_manifest, test_traj,
                             make_predictor(model_name, model, scaler, window=window))
    pred_rows = pd.DataFrame({"engine_id": engines, "true_rul_official": rul.astype(float),
                              "prediction": pred})
    pred_rows.to_csv(OUT_DIR / "fd001_official_predictions.csv", index=False)

    # independent recomputation (falsification 3b)
    y_true = rul.astype(float)
    metrics = {
        "methodology_version": "2.2",
        "dataset": "FD001",
        "candidate": candidate,
        "official_status": "post-hoc (FD001 official labels permanently post-hoc; never select or tune V2.2)",
        "official_test_engine_count": int(len(engines)),
        "official_test_RMSE": round(float(rmse(y_true, pred)), 4),
        "official_test_MAE": round(float(mae(y_true, pred)), 4),
        "official_test_R2": round(float(r2(y_true, pred)), 4),
        "official_test_NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "official_test_NASA_mean": round(float(nasa_score(y_true, pred)) / len(engines), 4),
        "cv_RMSE_mean_std": f"{cfg['selection']['cv_metrics']['RMSE_mean']} +- "
                            f"{cfg['selection']['cv_metrics']['RMSE_std']}",
    }
    Path(OUT_DIR / "fd001_final_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv("reports/tables/v2_2_fd001_official.csv", index=False)
    pd.DataFrame({"engine_id": engines, "true_rul_official": y_true,
                  "prediction": pred}).to_csv(
        "reports/tables/v2_2_fd001_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()