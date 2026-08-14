"""Train a classical RUL baseline on engineered features and log validation metrics.

Evaluation is on VALIDATION ENGINES only (never official test data).

Usage:
    .venv/Scripts/python.exe scripts/run_experiment.py --dataset FD001 --model r xgboost
    (model in: mean, linear, rf, xgboost)
"""

from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rul_prediction.data.loader import load_train
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.features.engineered_features import extract_features
from rul_prediction.models.baseline import MeanBaseline, linear_regressor, random_forest
from rul_prediction.models.xgboost_model import xgboost_regressor

WINDOW = 30
SEED = 42
MAX_RUL = 125
PROCESSED = Path("data/processed")
RESULTS_CSV = Path("experiments/results.csv")

MODELS = {"mean": MeanBaseline, "linear": linear_regressor, "rf": random_forest, "xgboost": xgboost_regressor}


def _load_windows(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    train_d = np.load(PROCESSED / f"{dataset}_train_sequences.npz")
    val_d = np.load(PROCESSED / f"{dataset}_validation_sequences.npz")
    lifetimes = load_train(dataset)["cycle"].groupby(load_train(dataset)["engine_id"]).max().to_dict()
    return train_d, val_d, lifetimes


def _final_cycles(engine_ids: np.ndarray, lifetimes: dict) -> np.ndarray:
    """Final cycle of each window; make_sequences orders windows in engine blocks, cycles ascending."""
    out = np.empty(len(engine_ids), dtype=int)
    i = 0
    for engine, block_len in _block_lengths(engine_ids).items():
        out[i : i + block_len] = WINDOW + np.arange(block_len)
        i += block_len
    return out


def _block_lengths(engine_ids: np.ndarray) -> dict:
    lengths, cur, count = {}, engine_ids[0], 0
    for e in engine_ids:
        if e == cur:
            count += 1
        else:
            lengths[cur] = count
            cur, count = e, 1
    lengths[cur] = count
    return lengths


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical RUL baseline runner")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--model", required=True, choices=list(MODELS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--notes", default="window-level engineered features")
    args = parser.parse_args()

    train_d, val_d, lifetimes = _load_windows(args.dataset)
    Xtr, ytr, ids_tr = train_d["X"], train_d["y"], train_d["engine_ids"]
    Xva, yva, ids_va = val_d["X"], val_d["y"], val_d["engine_ids"]

    Ftr, names = extract_features(Xtr, _final_cycles(ids_tr, lifetimes))
    Fva, _ = extract_features(Xva, _final_cycles(ids_va, lifetimes))
    feature_desc = f"engineered:{len(names)}"

    t0 = time.monotonic()
    if args.model == "mean":
        model = MeanBaseline().fit(Ftr, ytr)
    elif args.model == "xgboost":
        model = xgboost_regressor(args.seed)
        model.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)
    else:
        model = MODELS[args.model](args.seed)
        model.fit(Ftr, ytr)
    train_time = time.monotonic() - t0

    pred = np.clip(np.asarray(model.predict(Fva), dtype=float), 0, MAX_RUL)
    metrics = {
        "rmse": rmse(yva, pred),
        "mae": mae(yva, pred),
        "r2": r2(yva, pred),
        "nasa": nasa_score(yva, pred),
    }

    experiment_id = f"{args.dataset}_{args.model}_seed{args.seed}_{int(time.time())}"
    row = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": args.dataset,
        "model": args.model,
        "seed": args.seed,
        "features": feature_desc,
        "RUL cap": MAX_RUL,
        "validation RMSE": round(metrics["rmse"], 4),
        "validation MAE": round(metrics["mae"], 4),
        "validation R2": round(metrics["r2"], 4),
        "NASA score": round(metrics["nasa"], 3),
        "training time (s)": round(train_time, 3),
        "notes": args.notes,
    }

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[{args.model}] features={len(names)}  train_time={train_time:.2f}s")
    print(f"  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}  NASA={metrics['nasa']:.3f}")
    print(f"  logged -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()