"""Methodology V2.1: engine-cluster conformal calibration (FD001).

Calibration unit = ENGINE (not checkpoint row): for each of the 15 calibration
engines, the nonconformity score is the max |error| over its five fixed
checkpoints (0.25/0.45/0.65/0.80/0.95). With exchangeable engines under the
predefined lifecycle-checkpoint scheme, simultaneous coverage >= 1 - alpha is
guaranteed for the k-th smallest score with k = ceil((n+1)(1-alpha)) clamped
to [1, n].

The frozen model (models/v2_1/fd001_gru_w45_huber.keras) is evaluated on the
calibration manifest; intervals for held-out engines are reported empirically
(development folds + official test) separately from the formal guarantee.

Outputs:
    reports/tables/v2_1_conformal_calibration.csv   (15 engine scores + q)
    reports/tables/v2_1_conformal_coverage.csv      (held-out empirical coverage)
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

ALPHAS = (0.1, 0.2, 0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="V2.1 engine-level conformal calibration")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    args = parser.parse_args()

    model = keras.models.load_model(ROOT / "models/v2_1/fd001_gru_w45_huber.keras")
    scaler = load_joblib(ROOT / "models/v2_1/fd001_scaler.joblib")
    window = 45
    predictor = make_predictor("gru", model, scaler, window)

    frame = load_train("FD001", args.data_dir)
    cal_ids = pd.read_csv(Path(args.splits_dir) / "fd001_v2_1_calibration_cutoffs.csv")
    trajectories = {
        int(e): g.sort_values("cycle")
        for e, g in frame[frame["engine_id"].isin(cal_ids["engine_id"].unique())]
        .groupby("engine_id")
    }
    cal_manifest = load_manifest(Path(args.splits_dir) / "fd001_v2_1_calibration_cutoffs.csv")
    errors = np.zeros((len(cal_manifest),))
    pred = evaluate_manifest(cal_manifest, trajectories, predictor)
    errors = pred - cal_manifest["true_raw_rul"].to_numpy(dtype=float)
    err_df = pd.DataFrame({
        "engine_id": cal_manifest["engine_id"],
        "fraction": cal_manifest["fraction"],
        "error": errors,
    })
    scores = engine_cluster_scores(
        err_df.pivot(index="engine_id", columns="fraction", values="error").to_numpy())
    score_rows = [
        {"engine_id": int(e), "max_abs_error": float(s), "n_checkpoints": 5}
        for e, s in zip(err_df["engine_id"].unique(), scores)
    ]
    assert len(score_rows) == 15

    qs = {}
    ordered = np.sort(scores)
    for alpha in ALPHAS:
        k = finite_sample_quantile_index(15, alpha)
        qs[alpha] = quantile_from_index(ordered, k)
    print(f"engine-level scores (n=15): {np.round(ordered, 2)}")
    print(f"q(alpha): " + ", ".join(f"alpha={a}: {q:.2f} (k={finite_sample_quantile_index(15, a)})"
                                    for a, q in qs.items()))

    cal_err = err_df.groupby("engine_id")["error"].apply(
        lambda g: g.abs().max()).to_numpy()
    print(f"calibration coverage (by construction): "
          f"{np.mean(cal_err <= qs[0.1]) * 100:.1f}% at alpha=0.1")

    test = load_test("FD001", args.data_dir)
    rul = load_rul("FD001", args.data_dir)
    test_traj = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}
    observed = {int(e): int(g["cycle"].max()) for e, g in test_traj.items()}
    test_manifest = pd.DataFrame({"engine_id": list(observed), "cutoff_cycle": list(observed.values())})
    test_pred = evaluate_manifest(test_manifest, test_traj, predictor)
    test_errors = test_pred - rul.astype(float)
    coverage_rows = []
    for alpha, q in qs.items():
        in_interval = np.abs(test_errors) <= q
        coverage_rows.append({
            "alpha": alpha, "q": round(q, 4), "n": 15,
            "k": finite_sample_quantile_index(15, alpha),
            "official_coverage": round(float(np.mean(in_interval)), 4),
            "official_coverage_padded_lt45": round(
                float(np.mean(in_interval[np.array([o < window for o in observed.values()])])), 4),
            "official_coverage_full_ge45": round(
                float(np.mean(in_interval[np.array([o >= window for o in observed.values()])])), 4),
        })
    print(pd.DataFrame(coverage_rows).to_string(index=False))

    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    with open("reports/tables/v2_1_conformal_calibration.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["engine_id", "max_abs_error", "n_checkpoints"])
        writer.writeheader()
        writer.writerows(score_rows)
    pd.DataFrame(coverage_rows).to_csv("reports/tables/v2_1_conformal_coverage.csv", index=False)
    pd.DataFrame({"alpha": list(qs), "q": list(qs.values())}).to_csv(
        "reports/tables/v2_1_conformal_q.csv", index=False)
    Path("experiments/v2_1/conformal_calibration.json").write_text(
        json.dumps({"n_scores": 15, "checkpoints": 5, "q_by_alpha": qs,
                    "scores": score_rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()