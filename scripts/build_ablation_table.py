"""Build reports/tables/ablation_results.csv from validation-only results.csv rows.

Rows are classified by parsing the variant + model + loss fields into
ablation factors (A window, B RUL cap, C loss, D sensors, E architecture).
The final configuration row is marked.  emits a compact, per-factor table.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

RESULTS_CSV = Path("experiments/results.csv")
OUT = Path("reports/tables/ablation_results.csv")

FINAL = ("w90_c45_all", "xgboost")


def classify(row: dict) -> list[str]:
    variant = row["variant"]
    w = int(re.search(r"w(\d+)", variant).group(1))
    cap = re.search(r"c(none|\d+)", variant).group(1)
    m = re.search(r"w\d+_c(none|\d+)_(\w+)", variant)
    sensors = m.group(2) if m else ""

    labels: list[str] = []
    if variant == "w90_c45_all" and row["model"] == "gru":
        labels.append("composed A+B")
    if w != 90 or variant == "w90_c125_all":
        labels.append(f"A window={w}")
    if cap != "125":
        labels.append(f"B cap={cap}")
    if row["model"] in ("lstm", "gru", "tcn") and row["loss"] != "mse":
        labels.append(f"C loss={row['loss']}")
    if sensors == "varying":
        labels.append("D sensors=varying")
    if variant == "w90_c45_all":
        labels.append("E architecture")
    return labels


def main() -> None:
    rows = []
    with RESULTS_CSV.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            variant = r["notes"].split()[0].split("=")[1] if "variant=" in r["notes"] else "w30_c125_all"
            r["variant"] = variant
            loss = re.search(r"loss=(\w+)", r["notes"])
            r["loss"] = loss.group(1) if loss else "-"
            labels = classify(r)
            if not labels:
                continue
            r["factor"] = "; ".join(labels)
            rows.append(r)

    ordered = ["factor", "model", "variant", "loss",
               "validation RMSE", "validation MAE", "validation R2",
               "NASA score", "training time (s)"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"wrote {len(rows)} rows -> {OUT}")
    for r in rows:
        print(f"  {r['factor'][:48]:48s} {r['model']:8s} rmse={r['validation RMSE']:>8s} nasa={r['NASA score']:>10s}")


if __name__ == "__main__":
    main()