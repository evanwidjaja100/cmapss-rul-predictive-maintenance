"""Methodology V2.2: engine-cluster conformal calibration on the CLEAN final model.

Runs only after the V2.2 final fit (scripts/run_v2_2_freeze.py) — the 15
calibration engines' labels are touched HERE, after fitting is complete
(V2_2_REPAIR_PLAN.md V2.2-1/V2.2-5). The V2.2 model fit and model selection
never used these engines. IMPORTANT historical caveat: these calibration
engines were inspected during EARLIER project iterations, so the resulting
interval is an empirically calibrated uncertainty interval rather than a
pristine one-shot external conformal guarantee.

Protocol:
    - 15 calibration engines, 5 fixed lifecycle checkpoints
      (0.25/0.45/0.65/0.80/0.95) -> 75 rows;
    - score_engine = max |prediction - true_rul| across the 5 cutoffs
      (exactly one score per engine, 15 scores total);
    - interval 1-alpha: k = ceil((n+1)(1-alpha)) clamped to [1, n], q = k-th
      ordered score.

Formal statement (only under exchangeability of engines + the predefined
checkpoint scheme). Use on arbitrary uploaded trajectories is an ENGINEERING
EXTRAPOLATION and is labeled as such in the app and docs.

Outputs:
    experiments/v2_2/fd001_conformal_engine_scores.csv
    experiments/v2_2/fd001_conformal_quantiles.csv
    reports/tables/v2_2_conformal_*.csv
    experiments/v2_2/conformal_calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load as load_joblib
from tensorflow import keras

from rul_prediction.benchmark.v2 import ROOT, make_predictor
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.evaluation.conformal import (
    engine_cluster_scores,
    finite_sample_quantile_index,
    quantile_from_index,
)
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.data.loader import load_test, load_rul, load_train

OUT_DIR = Path("experiments/v2_2")
SPLITS_DIR = Path("experiments/splits")
ALPHAS = (0.1, 0.2, 0.3)
N_CALIBRATION_ENGINES = 15
N_CHECKPOINTS = 5


def calibration_scores(model, scaler, window: int, candidate: str) -> pd.DataFrame:
    """15 engine-level scores: max |error| over the 5 checkpoints per engine."""
    model_name = candidate.split("_")[0]
    model_name = "xgboost" if model_name == "xgb" else model_name
    predictor = make_predictor(model_name, model, scaler, window)
    frame = load_train("FD001")
    cal = pd.read_csv(SPLITS_DIR / "fd001_v2_2_calibration_cutoffs.csv")
    cal_ids = set(cal["engine_id"].unique())
    assert len(cal_ids) == N_CALIBRATION_ENGINES
    trajectories = {
        int(e): g.sort_values("cycle")
        for e, g in frame[frame["engine_id"].isin(cal_ids)].groupby("engine_id")
    }
    manifest = load_manifest(SPLITS_DIR / "fd001_v2_2_calibration_cutoffs.csv")
    assert len(manifest) == N_CALIBRATION_ENGINES * N_CHECKPOINTS
    errors = evaluate_manifest(manifest, trajectories, predictor) - \
        manifest["true_raw_rul"].to_numpy(dtype=float)
    err_df = pd.DataFrame({"engine_id": manifest["engine_id"],
                           "fraction": manifest["fraction"], "error": errors})
    scores = engine_cluster_scores(
        err_df.pivot(index="engine_id", columns="fraction", values="error").to_numpy())
    assert scores.shape == (N_CALIBRATION_ENGINES,)
    return pd.DataFrame({"engine_id": sorted(cal_ids),
                         "max_abs_error": np.round(scores, 4)})


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.2 conformal calibration")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--candidate", default=None,
                        help="candidate id from the final YAML (default: read from config)")
    args = parser.parse_args()

    import yaml
    cfg = yaml.safe_load(Path("configs/final_model_v2_2_fd001.yaml").read_text(encoding="utf-8"))
    candidate = args.candidate or cfg["model"]["candidate_name"]
    window = cfg["model"]["window"]
    arch = candidate.split("_")[0]
    if arch.startswith(("gru", "lstm")):
        model_path = ROOT / "models" / "v2_2" / f"fd001_{candidate}.keras"
        model = keras.models.load_model(model_path)
    else:
        from joblib import load as load_joblib
        model = load_joblib(ROOT / "models" / "v2_2" / f"fd001_{candidate}.joblib")
    scaler = load_joblib(ROOT / "models" / "v2_2" / "fd001_scaler.joblib")

    scores_df = calibration_scores(model, scaler, window, candidate)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(OUT_DIR / "fd001_conformal_engine_scores.csv", index=False)

    ordered = np.sort(scores_df["max_abs_error"].to_numpy())
    qs = {}
    for alpha in ALPHAS:
        k = finite_sample_quantile_index(N_CALIBRATION_ENGINES, alpha)
        qs[str(alpha)] = round(quantile_from_index(ordered, k), 4)
    pd.DataFrame({"alpha": list(qs), "q": list(qs.values())}).to_csv(
        OUT_DIR / "fd001_conformal_quantiles.csv", index=False)
    print(f"n scores = {len(scores_df)}; ordered = {np.round(ordered, 2)}")
    print(f"q(alpha): " + ", ".join(f"alpha={a}: {q}" for a, q in qs.items()))

    # ---- post-hoc empirical coverage on official FD001 (separate from guarantee) ----
    test = load_test("FD001", args.data_dir)
    rul = load_rul("FD001", args.data_dir)
    test_traj = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}
    observed = {int(e): int(g["cycle"].max()) for e, g in test_traj.items()}
    test_manifest = pd.DataFrame({"engine_id": list(observed),
                                  "cutoff_cycle": list(observed.values())})
    pred = evaluate_manifest(test_manifest, test_traj, make_predictor(
        "xgboost" if arch == "xgb" else arch, model, scaler, window))
    test_errors = pred - rul.astype(float)
    coverage_rows = []
    for alpha, q in qs.items():
        in_interval = np.abs(test_errors) <= q
        coverage_rows.append({
            "alpha": alpha, "q": q, "n": N_CALIBRATION_ENGINES,
            "k": finite_sample_quantile_index(N_CALIBRATION_ENGINES, float(alpha)),
            "official_coverage": round(float(np.mean(in_interval)), 4),
            "official_coverage_full_history": round(
                float(np.mean(in_interval[np.array([o >= window for o in observed.values()])])), 4),
            "official_coverage_short_history": round(
                float(np.mean(in_interval[np.array([o < window for o in observed.values()])])), 4),
        })
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv("reports/tables/v2_2_conformal_coverage.csv", index=False)
    print(pd.DataFrame(coverage_rows).to_string(index=False))

    result = {
        "methodology": "v2.2",
        "candidate": candidate,
        "n_calibration_engines": N_CALIBRATION_ENGINES,
        "n_checkpoints_per_engine": N_CHECKPOINTS,
        "n_scores": int(len(scores_df)),
        "score_definition": "max |prediction - true_rul| across 5 fixed lifecycle checkpoints",
        "q_by_alpha": {k: float(v) for k, v in qs.items()},
        "scores": scores_df.to_dict("records"),
        "formal_wording": ("Engine-cluster conformal interval calibrated using one maximum-error "
                           "score per held-out calibration engine across five predefined lifecycle "
                           "checkpoints; simultaneous coverage >= 1-alpha holds under exchangeability "
                           "of engines with the predefined checkpoint scheme."),
        "historical_caveat": ("The calibration engines were held out from V2.2 fitting and model "
                              "selection but were inspected during earlier project iterations; the "
                              "interval is an empirically calibrated uncertainty interval, not a "
                              "pristine one-shot external conformal guarantee."),
        "engineering_extrapolation": ("Use on arbitrary uploaded trajectories is an engineering "
                                      "extrapolation, not a formal guarantee."),
        "post_hoc_official_coverage": coverage_rows,
    }
    Path(OUT_DIR / "conformal_calibration.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()