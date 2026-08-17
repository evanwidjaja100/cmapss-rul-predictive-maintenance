"""Methodology V2.2: sensor sensitivity / occlusion / counterfactual attribution.

Reruns sensitivity analysis on the FINAL V2.2 deployment model (the V2 script
targeted the old V2 model; V2.2-9). Terminology: sensor sensitivity, sensor
occlusion, counterfactual attribution. These are NOT SHAP values.

Method (descriptive, post-freeze analysis on development engines only):
    For each sensor s: occlude the sensor's observed values with that engine's
    own per-engine mean (neutral in-range counterfactual) and measure the
    prediction change across the 5 fixed lifecycle checkpoints of the outer
    pseudo-test manifests.

Answers:
    - Which sensors most affect V2.2 predictions?  (mean |delta prediction|)
    - Do sensitivities change with true-RUL region? (per-fraction deltas)
    - Which sensors affect dangerous overprediction? (share of checkpoints
      where occlusion increases (pred - true))

Outputs:
    reports/v2_2_sensitivity.md
    reports/tables/v2_2_sensor_sensitivity.csv
    reports/tables/v2_2_temporal_sensitivity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import load as load_joblib
from tensorflow import keras

from rul_prediction.benchmark.v2 import ROOT, make_predictor
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.evaluation.manifest import evaluate_manifest

OUT_DIR = Path("reports/tables")


def load_dev_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Concatenated outer-fold pseudo-test manifests (development engines only)."""
    manifests, trajectories = [], {}
    for fold in range(1, 6):
        m = load_manifest(Path("experiments/splits") /
                          f"fd001_v2_2_outer_fold{fold}_cutoffs.csv")
        m["fold"] = fold
        manifests.append(m)
        for e, g in frame[frame["engine_id"].isin(m["engine_id"])].groupby("engine_id"):
            trajectories[int(e)] = g.sort_values("cycle")
    return pd.concat(manifests, ignore_index=True), trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.2 sensor sensitivity")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/final_model_v2_2_fd001.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidate = cfg["model"]["candidate_name"]
    window = cfg["model"]["window"]
    arch = candidate.split("_")[0]
    model_name = "xgboost" if arch == "xgb" else arch

    if model_name in ("rf", "xgboost"):
        model = load_joblib(ROOT / "models" / "v2_2" / f"fd001_{candidate}.joblib")
    else:
        model = keras.models.load_model(ROOT / "models" / "v2_2" / f"fd001_{candidate}.keras")
    scaler = load_joblib(ROOT / "models" / "v2_2" / "fd001_scaler.joblib")
    predict_one = make_predictor(model_name, model, scaler, window)

    frame = load_train("FD001", args.data_dir)
    manifest, trajectories = load_dev_manifest(frame)
    engine_means = frame.groupby("engine_id")[SENSOR_COLUMNS].mean()

    baseline = evaluate_manifest(manifest, trajectories, predict_one)
    err_base = baseline - manifest["true_raw_rul"].to_numpy()
    baseline_by_key = {(int(r.engine_id), int(r.cutoff_cycle)): float(b)
                       for r, b in zip(manifest.itertuples(index=False), baseline)}

    rows, temporal_rows = [], []
    for sensor in SENSOR_COLUMNS:
        deltas = []
        for e, g in manifest.groupby("engine_id"):
            history = trajectories[int(e)].copy()
            history[sensor] = float(engine_means.loc[e, sensor])
            for _, r in g.iterrows():
                cutoff = int(r["cutoff_cycle"])
                p_occ = predict_one(history[history["cycle"] <= cutoff], cutoff)
                deltas.append({"engine_id": int(r.engine_id), "fraction": r["fraction"],
                               "true_raw_rul": r["true_raw_rul"],
                               "prediction_baseline": baseline_by_key[(int(r.engine_id), cutoff)],
                               "prediction_occluded": float(p_occ)})
        d = pd.DataFrame(deltas)
        delta = d["prediction_occluded"] - d["prediction_baseline"]
        d["delta"] = delta
        rows.append({
            "sensor": sensor,
            "mean_abs_delta": round(float(np.abs(delta).mean()), 4),
            "rmse_delta": round(float(np.sqrt(np.mean((err_base + delta) ** 2)) -
                                      np.sqrt(np.mean(err_base ** 2))), 4),
            "overprediction_worsened_share": round(
                float(np.mean(delta > 0)), 4),
            "overprediction_share_among_dangerous": round(float(np.mean(
                (d["prediction_baseline"] > d["true_raw_rul"]) & (delta > 0))), 4),
        })
        for frac, g in d.groupby("fraction"):
            temporal_rows.append({
                "sensor": sensor, "fraction": float(frac),
                "mean_abs_delta": round(float(np.abs(g["delta"]).mean()), 4),
                "mean_delta": round(float(g["delta"].mean()), 4),
            })

    sens = pd.DataFrame(rows).sort_values("mean_abs_delta", ascending=False)
    temporal = pd.DataFrame(temporal_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sens.to_csv(OUT_DIR / "v2_2_sensor_sensitivity.csv", index=False)
    temporal.to_csv(OUT_DIR / "v2_2_temporal_sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    md_lines = [
        f"# V2.2 sensor sensitivity (final model: {candidate})",
        "",
        "Method: per-sensor occlusion (sensor values replaced by the engine's own",
        "mean over its observed history) on the fixed outer pseudo-test manifests",
        f"of the 85 development engines ({len(manifest)} checkpoints, 5 fractions).",
        "Descriptive post-freeze analysis; NOT SHAP values; not used for selection.",
        "",
        "| Sensor | mean |delta| | RMSE delta | overprediction-worsened share |",
        "|---|---|---|---|",
    ]
    for _, r in sens.iterrows():
        md_lines.append(f"| {r['sensor']} | {r['mean_abs_delta']:.3f} | "
                        f"{r['rmse_delta']:+.3f} | {r['overprediction_worsened_share']:.3f} |")
    md_lines += ["", "## Region (fraction) sensitivity — top-3 sensors per region", ""]
    for frac, g in temporal.groupby("fraction"):
        top = g.sort_values("mean_abs_delta", ascending=False).head(3)
        md_lines.append(f"fraction {frac:.2f}: " +
                        ", ".join(f"{r['sensor']} ({r['mean_abs_delta']:.3f})" for _, r in top.iterrows()))
    Path("reports/v2_2_sensitivity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("wrote reports/v2_2_sensitivity.md")


if __name__ == "__main__":
    main()