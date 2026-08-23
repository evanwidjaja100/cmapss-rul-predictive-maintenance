"""Methodology M3: error analysis of the frozen M3 FD001 model (post-hoc).

Descriptive analysis on the POST-HOC official predictions only — never used
for model selection, training control, or SERVING behavior. Reports the
signed-bias profile by observed-history length. No threshold from this
analysis is used by the Streamlit app: serving exposes only objective
padding/history facts.

Outputs:
    reports/m3_error_analysis.md
    reports/tables/m3_error_profile.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rul_prediction.data.loader import load_test

OUT_DIR = Path("reports/tables")


def history_bucket_upper(observed_bucket: str) -> int:
    """Numeric upper bound of an observed bucket label like '[0,45)' (never lexicographic)."""
    lo, hi = observed_bucket.strip("[]()").split(",")
    return int(hi)


def main() -> None:
    test = load_test("FD001")
    observed = test.groupby("engine_id")["cycle"].max().rename("observed_cycles")
    pred = pd.read_csv("experiments/m3/fd001_official_predictions.csv")
    frame = pred.merge(observed, on="engine_id")
    frame["error"] = frame["prediction"] - frame["true_rul_official"]
    frame["abs_error"] = frame["error"].abs()

    buckets = [(0, 45), (45, 90), (90, 128), (128, 200), (200, 10_000)]
    rows = []
    for lo, hi in buckets:
        g = frame[(frame["observed_cycles"] >= lo) & (frame["observed_cycles"] < hi)]
        if len(g) == 0:
            continue
        rows.append({
            "observed_bucket": f"[{lo},{hi})",
            "n_engines": int(len(g)),
            "mean_signed_error": round(float(g["error"].mean()), 3),
            "overprediction_share": round(float((g["error"] > 0).mean()), 3),
            "mean_abs_error": round(float(g["abs_error"].mean()), 3),
        })
    profile = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_csv(OUT_DIR / "m3_error_profile.csv", index=False)
    print(profile.to_string(index=False))

    md = [
        "# M3 error analysis (frozen model, POST-HOC official FD001)",
        "",
        "The official FD001 labels are permanently post-hoc (inspected in the M1-0",
        "audit). This analysis is descriptive; it never selects or tunes the model.",
        "",
        f"Overall: mean signed error {frame['error'].mean():+.2f} cycles; "
        f"overprediction share {(frame['error'] > 0).mean():.2f}.",
        "",
        "| observed bucket | n engines | mean signed error | overprediction share | mean abs error |",
        "|---|---|---|---|---|",
    ]
    for _, r in profile.iterrows():
        md.append(f"| {r['observed_bucket']} | {int(r['n_engines'])} | "
                  f"{r['mean_signed_error']:+.3f} | {r['overprediction_share']:.3f} | "
                  f"{r['mean_abs_error']:.3f} |")
    short = profile[profile["observed_bucket"] != "[200,10000)"]
    threshold = max(history_bucket_upper(b) for b in short["observed_bucket"])
    md += [
        "",
        f"Descriptive observation: among observed-history buckets below "
        f"{threshold} cycles, signed errors and absolute errors are larger. "
        "This is a post-hoc descriptive finding; it does NOT drive serving "
        "behavior (the app exposes only objective padding/history facts).",
    ]
    Path("reports/m3_error_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"descriptive short-history observation: observed < {threshold} cycles")


if __name__ == "__main__":
    main()