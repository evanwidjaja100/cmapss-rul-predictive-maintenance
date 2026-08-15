"""Methodology V2 controlled ablations (target definition is NEVER changed).

Allowed: window length, model depth, number of estimators, dropout, loss.
Forbidden: RUL cap - the raw-RUL target is fixed; the legacy cap sweep is
historical only.

Every run evaluates on the SAME fixed 75-row validation pseudo-test manifest,
so NASA totals are directly comparable. All decisions use validation results
only; the official FD001 test set is NOT contacted.

Results append to experiments/v2_results.csv (idempotent) and the curated
table is written to reports/tables/v2_ablation_results.csv.

Usage:
    .venv/Scripts/python.exe scripts/run_v2_ablation.py            (full matrix)
    .venv/Scripts/python.exe scripts/run_v2_ablation.py --window 15 --models rf xgboost
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from rul_prediction.benchmark.v2 import (
    ROOT,
    append_results,
    run_experiment,
)

WINDOWS_ALL = (15, 30, 45, 60, 90)      # classical models: full sweep
WINDOWS_DEEP = (15, 30, 45, 60, 90)     # deep models: full sweep (w45+ trend motivates 60/90)
CLASSICAL_MODELS = ("mean", "linear", "rf", "xgboost")
DEEP_MODELS = ("lstm", "gru", "tcn")


def build_matrix(only_windows=None, only_models=None) -> list[tuple[str, int, dict]]:
    """(model, window, overrides) triples; None = default architecture parameters."""
    runs = []
    for w in WINDOWS_ALL:
        if only_windows and w not in only_windows:
            continue
        for m in CLASSICAL_MODELS:
            if only_models and m not in only_models:
                continue
            runs.append((m, w, {}))
    for w in WINDOWS_DEEP:
        if only_windows and w not in only_windows:
            continue
        for m in DEEP_MODELS:
            if only_models and m not in only_models:
                continue
            runs.append((m, w, {}))
    # Hyperparameter ablations (target definition NEVER changed)
    if not only_windows or 30 in only_windows:
        if not only_models or "xgboost" in only_models:
            runs += [("xgboost", 30, {"max_depth": d}) for d in (3, 9)]
            runs += [("xgboost", 30, {"n_estimators": 300})]
        if not only_models or "lstm" in only_models:
            runs += [("lstm", 30, {"dropout": d}) for d in (0.2, 0.4)]
            runs += [("lstm", 30, {"loss": "huber"})]
    if (not only_windows or 90 in only_windows) and (not only_models or "xgboost" in only_models):
        runs += [("xgboost", 90, {"max_depth": 9})]
    if not only_models or "gru" in only_models:
        if (not only_windows or 45 in only_windows):
            runs += [("gru", 45, {"loss": "huber"})]
        if (not only_windows or 60 in only_windows):
            runs += [("gru", 60, {"loss": "huber"})]
    if (not only_windows or 45 in only_windows) and (not only_models or "lstm" in only_models):
        runs += [("lstm", 45, {"loss": "huber"})]
    return runs


def _experiment_id(model: str, window: int, overrides: dict) -> str:
    suffix = "".join(f"_{k}{v}" for k, v in sorted(overrides.items()))
    return f"v2_{model}_w{window}{suffix}_s42"


def main() -> None:
    parser = argparse.ArgumentParser(description="Methodology V2 controlled ablations")
    parser.add_argument("--window", type=int, nargs="*", help="restrict to these windows")
    parser.add_argument("--models", nargs="*", help="restrict to these models")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    out_csv = ROOT / "experiments" / "v2_results.csv"
    pred_csv = ROOT / "experiments" / "v2_validation_predictions.csv"
    done_ids = set()
    if out_csv.exists():
        done_ids = {r["experiment_id"] for r in csv.DictReader(out_csv.open(encoding="utf-8"))}

    matrix = build_matrix(only_windows=set(args.window) if args.window else None,
                          only_models=set(args.models) if args.models else None)
    print(f"ablation matrix: {len(matrix)} runs ({sum(1 for m, _, _ in matrix if m not in DEEP_MODELS)} classical + "
          f"{sum(1 for m, _, _ in matrix if m in DEEP_MODELS)} deep)")
    rows, predictions = [], []
    for model_name, window, overrides in matrix:
        experiment_id = _experiment_id(model_name, window, overrides)
        if experiment_id in done_ids:
            print(f"skipping {experiment_id} (already present)")
            continue
        row, prediction_rows = run_experiment(
            experiment_id, model_name, window, overrides,
            dataset=args.dataset, data_dir=args.data_dir, splits_dir=args.splits_dir,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
        rows.append(row)
        predictions.extend(prediction_rows)
        print(f"{model_name:8s} w{window:>3d} {str(overrides or ''):24s} "
              f"RMSE={row['RMSE']:8.4f} MAE={row['MAE']:8.4f} R2={row['R2']:8.4f} "
              f"NASA_total={row['NASA_total']:9.2f} time={row['training_time']}s")
    if rows:
        append_results(out_csv, pred_csv, rows, predictions)

    # Curated ablation table (tracked in reports/tables/)
    table = ROOT / "reports" / "tables" / "v2_ablation_results.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    keys = ["experiment_id", "model", "window", "seed", "validation_sample_count",
            "RMSE", "MAE", "R2", "NASA_total", "NASA_mean", "training_time",
            "feature_count", "parameters", "notes"]
    with table.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        with out_csv.open(encoding="utf-8") as src:
            for r in csv.DictReader(src):
                writer.writerow({k: r.get(k, "") for k in keys})
    print(f"\nwrote -> {table}")
    print("official FD001 test set NOT contacted")


if __name__ == "__main__":
    main()