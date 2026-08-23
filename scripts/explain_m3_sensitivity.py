"""Methodology M3: sensor sensitivity / occlusion / counterfactual attribution.

Reruns sensitivity analysis on the FINAL M3 deployment model (the M1 script
targeted the old M1 model; M3-9). Terminology: sensor sensitivity, sensor
occlusion, counterfactual attribution. These are NOT SHAP values.

Method (descriptive, post-freeze analysis on development engines only):
    For each sensor s and each fixed lifecycle checkpoint of the outer
    pseudo-test manifests: occlude the sensor with the engine's PREFIX-ONLY
    observed mean (mean over cycles <= cutoff; never future rows) and measure
    the prediction change. All rows are keyed by (engine_id, cutoff_cycle);
    RMSE deltas are computed from exactly aligned rows, never from accidental
    groupby ordering.

Answers:
    - Which sensors most affect M3 predictions?  (mean |delta prediction|)
    - Do sensitivities change with true-RUL region? (per-fraction deltas)
    - Which sensors affect dangerous overprediction? (share of checkpoints
      where occlusion increases (pred - true))

Outputs:
    reports/m3_sensitivity.md
    reports/tables/m3_sensor_sensitivity.csv
    reports/tables/m3_temporal_sensitivity.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import load as load_joblib
from tensorflow import keras

from rul_prediction.benchmark.m1 import ROOT, make_predictor
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.evaluation.manifest import evaluate_manifest

OUT_DIR = Path("reports/tables")


def load_dev_manifest(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Concatenated outer-fold pseudo-test manifests (development engines only)."""
    manifests, trajectories = [], {}
    for fold in range(1, 6):
        m = load_manifest(Path("experiments/splits") /
                          f"fd001_m3_outer_fold{fold}_cutoffs.csv")
        m["fold"] = fold
        manifests.append(m)
        for e, g in frame[frame["engine_id"].isin(m["engine_id"])].groupby("engine_id"):
            trajectories[int(e)] = g.sort_values("cycle")
    return pd.concat(manifests, ignore_index=True), trajectories


def prefix_replacement_value(history: pd.DataFrame, sensor: str, cutoff: int) -> float:
    """Prefix-only sensor baseline: mean over cycles <= cutoff (never future rows)."""
    observed = history[history["cycle"] <= cutoff]
    assert len(observed) > 0, f"no observed rows at cutoff {cutoff}"
    return float(observed[sensor].mean())


def sensor_occlusion_deltas(manifest: pd.DataFrame, trajectories: dict,
                            predict_one, sensor: str) -> pd.DataFrame:
    """Keyed (engine_id, cutoff_cycle) occlusion deltas for one sensor.

    The replacement value uses only cycles <= cutoff; row alignment is
    explicit via the key columns, never positional/groupby order.
    """
    rows = []
    for e, g in manifest.groupby("engine_id", sort=False):
        history = trajectories[int(e)]
        for _, r in g.iterrows():
            cutoff = int(r["cutoff_cycle"])
            occluded = history.copy()
            occluded[sensor] = prefix_replacement_value(history, sensor, cutoff)
            p_occ = predict_one(occluded[occluded["cycle"] <= cutoff], cutoff)
            rows.append({"engine_id": int(r.engine_id), "cutoff_cycle": cutoff,
                         "prediction_occluded": float(p_occ)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 sensor sensitivity")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/final_model_m3_fd001.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    candidate = cfg["model"]["candidate_name"]
    window = cfg["model"]["window"]
    arch = candidate.split("_")[0]
    model_name = "xgboost" if arch == "xgb" else arch

    if model_name in ("rf", "xgboost"):
        model = load_joblib(ROOT / "models" / "m3" / f"fd001_{candidate}.joblib")
    else:
        model = keras.models.load_model(ROOT / "models" / "m3" / f"fd001_{candidate}.keras")
    scaler = load_joblib(ROOT / "models" / "m3" / "fd001_scaler.joblib")
    predict_one = make_predictor(model_name, model, scaler, window=window)

    frame = load_train("FD001", args.data_dir)
    manifest, trajectories = load_dev_manifest(frame)

    baseline = evaluate_manifest(manifest, trajectories, predict_one)
    base = manifest[["engine_id", "cutoff_cycle", "fraction", "true_raw_rul"]].copy()
    base["prediction_baseline"] = baseline
    base["baseline_error"] = base["prediction_baseline"] - base["true_raw_rul"]

    rows, temporal_rows = [], []
    for sensor in SENSOR_COLUMNS:
        d = sensor_occlusion_deltas(manifest, trajectories, predict_one, sensor)
        merged = base.merge(d, on=["engine_id", "cutoff_cycle"], how="inner",
                            validate="one_to_one")
        assert len(merged) == len(base), f"alignment lost for {sensor}"
        merged["delta"] = merged["prediction_occluded"] - merged["prediction_baseline"]
        merged["occluded_error"] = merged["prediction_occluded"] - merged["true_raw_rul"]
        rmse_base = float(np.sqrt(np.mean(merged["baseline_error"] ** 2)))
        rmse_occ = float(np.sqrt(np.mean(merged["occluded_error"] ** 2)))
        rows.append({
            "sensor": sensor,
            "mean_abs_delta": round(float(np.abs(merged["delta"]).mean()), 4),
            "rmse_delta": round(rmse_occ - rmse_base, 4),
            "overprediction_worsened_share": round(
                float(np.mean(merged["delta"] > 0)), 4),
            "overprediction_share_among_dangerous": round(float(np.mean(
                (merged["prediction_baseline"] > merged["true_raw_rul"]) &
                (merged["delta"] > 0))), 4),
        })
        for frac, g in merged.groupby("fraction"):
            temporal_rows.append({
                "sensor": sensor, "fraction": float(frac),
                "mean_abs_delta": round(float(np.abs(g["delta"]).mean()), 4),
                "mean_delta": round(float(g["delta"].mean()), 4),
            })

    sens = pd.DataFrame(rows).sort_values("mean_abs_delta", ascending=False)
    temporal = pd.DataFrame(temporal_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sens.to_csv(OUT_DIR / "m3_sensor_sensitivity.csv", index=False)
    temporal.to_csv(OUT_DIR / "m3_temporal_sensitivity.csv", index=False)
    print(sens.to_string(index=False))

    md_lines = [
        f"# M3 sensor sensitivity (final model: {candidate})",
        "",
        "Method: per-sensor occlusion on the fixed outer pseudo-test manifests",
        f"of the 85 development engines ({len(manifest)} checkpoints, 5 fractions).",
        "The replacement value is the engine's PREFIX-ONLY observed mean (cycles",
        "<= cutoff; never future rows). Rows are aligned by (engine_id,",
        "cutoff_cycle); RMSE deltas use exactly aligned rows.",
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
    Path("reports/m3_sensitivity.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("wrote reports/m3_sensitivity.md")


if __name__ == "__main__":
    main()