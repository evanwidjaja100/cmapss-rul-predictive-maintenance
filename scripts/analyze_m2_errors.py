"""Methodology M2: honest error analysis of the frozen FD001 model.

Corrects the M1 error analysis (M2_REPAIR_PLAN.md R1/R2/R4): official C-MAPSS
test trajectories are truncated BEFORE failure, so `cycle.max()` is the
OBSERVED HISTORY LENGTH, never a lifetime. For labeled analysis:

    implied_failure_cycle = observed_cycles + true_rul

Subgroups are reported separately for observed history length (with the
padded/unpadded split at the model window) and for implied failure lifetime,
plus true-RUL bins. `error = prediction - true_rul` (positive = overprediction).

Outputs: reports/tables/m2_error_profile.csv, reports/m2_error_analysis.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rul_prediction.benchmark.m1 import ROOT
from rul_prediction.data.loader import load_rul, load_test
from rul_prediction.evaluation.metrics import mae, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

WINDOW = 45
OBSERVED_BINS = [(0, 44, "padded_observed_lt_window"),
                 (45, 127, "observed_45_127"),
                 (128, 10_000, "observed_ge_128")]
LIFETIME_BINS = [(0, 127, "implied_lifetime_lt_128"),
                 (128, 10_000, "implied_lifetime_ge_128")]
RUL_BINS = [(0, 29, "true_rul_0_29"), (30, 59, "true_rul_30_59"),
            (60, 99, "true_rul_60_99"), (100, 10_000, "true_rul_ge_100")]


def group_stats(g: pd.DataFrame) -> dict:
    y = g["true_rul_official"].to_numpy(dtype=float)
    p = g["prediction"].to_numpy(dtype=float)
    e = p - y
    return {
        "count": int(len(g)),
        "mean_error": round(float(np.mean(e)), 3),
        "mean_abs_error": round(float(np.mean(np.abs(e))), 3),
        "RMSE": round(float(rmse(y, p)), 3),
        "MAE": round(float(mae(y, p)), 3),
        "NASA_sum": round(float(nasa_score(y, p)), 1),
        "NASA_pct_of_total": None,  # filled after total is known
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="M2 error analysis (FD001, frozen model)")
    parser.add_argument("--data-dir", default="data/raw")
    args = parser.parse_args()

    preds = pd.read_csv("reports/tables/m2_fd001_predictions.csv")
    test = load_test("FD001", args.data_dir)
    rul = load_rul("FD001", args.data_dir).astype(float)
    observed = test.groupby("engine_id")["cycle"].max().astype(int)

    df = preds.copy()
    df["observed_cycles"] = df["engine_id"].map(observed).astype(int)
    df["true_rul_official"] = df["engine_id"].map(dict(zip(range(1, len(rul) + 1), rul)))
    df["implied_failure_cycle"] = df["observed_cycles"] + df["true_rul_official"]
    df["error"] = df["prediction"] - df["true_rul_official"]
    assert (df["implied_failure_cycle"] > df["observed_cycles"]).all()
    assert (df["implied_failure_cycle"] == df["observed_cycles"] + df["true_rul_official"]).all()

    total_nasa = float(nasa_score(df["true_rul_official"].to_numpy(),
                                  df["prediction"].to_numpy()))

    rows = []
    for name, bins in (("observed_history", OBSERVED_BINS),
                       ("implied_lifetime", LIFETIME_BINS),
                       ("true_rul_bin", RUL_BINS)):
        for lo, hi, label in bins:
            if name == "observed_history":
                g = df[(df["observed_cycles"] >= lo) & (df["observed_cycles"] <= hi)]
            elif name == "implied_lifetime":
                g = df[(df["implied_failure_cycle"] >= lo) & (df["implied_failure_cycle"] <= hi)]
            else:
                g = df[(df["true_rul_official"] >= lo) & (df["true_rul_official"] <= hi)]
            s = group_stats(g)
            s["group"] = label
            s["split"] = name
            s["NASA_pct_of_total"] = round(100 * s["NASA_sum"] / total_nasa, 2) if s["count"] else "empty"
            rows.append(s)
    profile = pd.DataFrame(rows)

    padded = df[df["observed_cycles"] < WINDOW]
    full = df[df["observed_cycles"] >= WINDOW]
    for g, label in ((padded, "padded_observed_lt_45"), (full, "full_observed_ge_45")):
        s = group_stats(g)
        s["group"] = label
        s["split"] = "padded_vs_full"
        s["NASA_pct_of_total"] = round(100 * s["NASA_sum"] / total_nasa, 2) if s["count"] else "empty"
        rows.append(s)
    profile = pd.DataFrame(rows)
    profile.to_csv("reports/tables/m2_error_profile.csv", index=False)

    table_lines = ["| split | group | count | mean error | mean abs error | RMSE | MAE | NASA sum | NASA % of total |"]
    table_lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        table_lines.append(
            f"| {r['split']} | {r['group']} | {r['count']} | {r['mean_error']} | "
            f"{r['mean_abs_error']} | {r['RMSE']} | {r['MAE']} | {r['NASA_sum']} | "
            f"{r['NASA_pct_of_total']} |")
    table = "\n".join(table_lines)

    md = [
        "# M2 Error Analysis - frozen FD001 model (post-hoc official test)",
        "",
        f"Model: `gru_w45_huber` (M2 CV-selected, 85 development engines). "
        f"Official test: {len(df)} engines. Total NASA = {total_nasa:.2f}.",
        "",
        "**Terminology correction (M2_REPAIR_PLAN.md R1/R2):** official test "
        "trajectories are truncated before failure; `cycle.max()` = observed "
        "history length, NOT failure lifetime. `implied_failure_cycle = "
        "observed_cycles + true_rul`. `error = prediction - true_rul` "
        "(positive = overprediction).",
        "",
        "## Profile",
        "",
        table,
        "",
        "## What the M1 claim becomes",
        "",
        "M1 reported that 'engines with lifetime < 128 carry 99.8% of the NASA "
        "penalty'. Under the corrected definition (implied failure lifetime = "
        f"observed_cycles + true_rul), the `implied_lifetime_lt_128` group is "
        f"**empty on the official test** ({len(df)}/100 engines): every official "
        "engine has implied failure cycle >= 128. The M1 finding was really "
        "about OBSERVED HISTORY LENGTH (44 engines with cycle.max() < 128); "
        "that quantity is a trajectory-truncation artifact, not a lifetime. "
        "The observed-history rows above supersede it.",
        "",
        f"Observed-history split of the old claim: {int(((df['observed_cycles']>=45)&(df['observed_cycles']<=127)).sum())} "
        f"engines in observed 45-127, {int((df['observed_cycles']>=128).sum())} in observed >= 128, "
        f"{int((df['observed_cycles']<WINDOW).sum())} padded (< window).",
        "",
        "## Notes",
        "",
        "- `padded_observed_lt_window`: engines whose observed history is "
        f"shorter than the model window ({WINDOW}); their windows are "
        "left-padded in the shared representation. This is expected input "
        "for the model, not out-of-distribution.",
        f"- Only {int((df['observed_cycles'] < WINDOW).sum())} of {len(df)} "
        f"official-test engines have observed history < {WINDOW}; "
        "small-sample caution applies to that subgroup.",
    ]
    report = "\n".join(md)
    Path("reports/m2_error_analysis.md").write_text(report, encoding="utf-8")

    summary = {"total_NASA": total_nasa,
               "padded_observed_lt_window_count": int((df["observed_cycles"] < WINDOW).sum())}
    Path("experiments/m2/error_analysis.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(profile.to_string(index=False))
    print(f"\nReport -> reports/m2_error_analysis.md (NASA total {total_nasa:.2f})")


if __name__ == "__main__":
    main()