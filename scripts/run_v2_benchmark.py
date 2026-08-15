"""Methodology V2 primary raw-RUL benchmark (window 30, no cap).

Runs mean/linear/random-forest/XGBoost/LSTM/GRU/TCN with IDENTICAL:
    - train partition (70 V2 engines), scaler (train-only), raw-RUL target
    - fixed 75-row validation pseudo-test manifest (every model, same rows)
Writes:
    experiments/v2_results.csv              (one row per model, idempotent)
    experiments/v2_validation_predictions.csv (raw predictions per manifest row)
The official FD001 test set is NOT contacted here.

Usage:
    .venv/Scripts/python.exe scripts/run_v2_benchmark.py
    .venv/Scripts/python.exe scripts/run_v2_benchmark.py --models lstm gru tcn
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rul_prediction.benchmark.v2 import (
    ROOT,
    append_results,
    run_experiment,
)

WINDOW = 30
MODELS = ("mean", "linear", "rf", "xgboost", "lstm", "gru", "tcn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Methodology V2 raw-RUL benchmark")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    out_csv = ROOT / "experiments" / "v2_results.csv"
    pred_csv = ROOT / "experiments" / "v2_validation_predictions.csv"
    done_ids = set()
    if out_csv.exists():
        import csv

        done_ids = {r["experiment_id"] for r in csv.DictReader(out_csv.open(encoding="utf-8"))}

    for model_name in args.models:
        experiment_id = f"v2_{model_name}_w{args.window}_s42"
        if experiment_id in done_ids:
            print(f"skipping {model_name} (already present as {experiment_id})")
            continue
        row, prediction_rows = run_experiment(
            experiment_id, model_name, args.window, None,
            dataset=args.dataset, data_dir=args.data_dir, splits_dir=args.splits_dir,
            epochs=args.epochs, batch_size=args.batch_size, patience=args.patience)
        append_results(out_csv, pred_csv, [row], prediction_rows)
        print(f"{model_name:8s} RMSE={row['RMSE']:8.4f} MAE={row['MAE']:8.4f} "
              f"R2={row['R2']:8.4f} NASA_total={row['NASA_total']:9.2f} "
              f"NASA_mean={row['NASA_mean']:8.4f} time={row['training_time']}s")
    print("official FD001 test set NOT contacted")


if __name__ == "__main__":
    main()