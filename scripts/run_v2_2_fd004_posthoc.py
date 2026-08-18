"""Methodology V2.2: post-hoc official FD004 evaluation (after YAML freeze).

Official FD004 labels are permanently POST-HOC (inspected in V2-11; never
select the variant). Falsification: variant results and headline metrics are
recomputed from the saved prediction CSV and compared with the config.
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

from rul_prediction.benchmark.v2 import ROOT
from rul_prediction.data.condition import condition_feature_matrix
from rul_prediction.data.loader import EXPECTED_ENGINE_COUNTS, load_rul, load_test
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.windows import build_window, window_mask
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

from run_v2_2_fd004 import build_matrix, make_predictor

OUT_DIR = Path("experiments/v2_2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc official FD004 evaluation")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/final_model_v2_2_fd004.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    variant = cfg["condition_preprocessing"]["variant"]
    window = cfg["model"]["window"]
    loss = cfg["model"]["loss"]

    model = keras.models.load_model(ROOT / "models" / "v2_2" /
                                    f"fd004_gru_w{window}_{loss}_cond{variant}.keras")
    cond = load_joblib(ROOT / "models" / "v2_2" / f"fd004_condition{variant}.joblib")
    global_scaler = None

    test = load_test("FD004", args.data_dir)
    rul = load_rul("FD004", args.data_dir).astype(float)
    engines = sorted(test["engine_id"].unique())
    assert len(engines) == len(rul) == EXPECTED_ENGINE_COUNTS["FD004"]["test"]
    test_manifest = pd.DataFrame({
        "engine_id": engines,
        "cutoff_cycle": [int(test[test["engine_id"] == e]["cycle"].max()) for e in engines]})
    test_traj = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}

    predictor = make_predictor(variant, model, cond["kmeans"], cond["cluster_scalers"],
                               cond["settings_scaler"], global_scaler)
    pred = evaluate_manifest(test_manifest, test_traj, predictor)
    y_true = np.asarray(rul, dtype=float)
    metrics = {
        "methodology_version": "2.2",
        "dataset": "FD004",
        "variant": variant,
        "official_status": "post-hoc (FD004 official labels permanently post-hoc; never select or tune V2.2)",
        "official_test_engine_count": int(len(engines)),
        "official_test_RMSE": round(float(rmse(y_true, pred)), 4),
        "official_test_MAE": round(float(mae(y_true, pred)), 4),
        "official_test_R2": round(float(r2(y_true, pred)), 4),
        "official_test_NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "official_test_NASA_mean": round(float(nasa_score(y_true, pred)) / len(engines), 4),
        "official_test_prediction_std": round(float(np.std(pred)), 4),
    }
    Path(OUT_DIR / "fd004_final_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame([metrics]).to_csv("reports/tables/v2_2_fd004_official.csv", index=False)
    pd.DataFrame({"engine_id": engines, "true_rul_official": y_true,
                  "prediction": pred}).to_csv(
        "reports/tables/v2_2_fd004_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))

    # ---- falsification: variant metrics recomputed from saved prediction CSV ----
    pred_df = pd.read_csv(OUT_DIR / "fd004_variant_predictions.csv")
    stored = pd.read_csv(OUT_DIR / "fd004_variant_results.csv").set_index("variant")
    for v, g in pred_df.groupby("variant"):
        y, p = g["true_raw_rul"].to_numpy(float), g["prediction"].to_numpy(float)
        re_rmse = round(float(rmse(y, p)), 4)
        re_nasa = round(float(nasa_score(y, p)), 2)
        assert abs(re_rmse - stored.loc[v, "RMSE"]) < 1e-3, f"variant {v} RMSE drift"
        assert abs(re_nasa - stored.loc[v, "NASA_total"]) < 1e-1, f"variant {v} NASA drift"
    print("falsification: stored FD004 variant metrics match saved predictions")


if __name__ == "__main__":
    main()