"""Train a baseline model and log validation metrics.

Classical models use engineered window features; lstm/gru consume raw
(scaled) windows directly. Evaluation is on VALIDATION ENGINES only.

Usage:
    .venv/Scripts/python.exe scripts/run_experiment.py --dataset FD001 --model xgboost
    .venv/Scripts/python.exe scripts/run_experiment.py --dataset FD001 --model lstm --loss huber
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

MODELS = {"mean", "linear", "rf", "xgboost", "lstm", "gru", "tcn"}


def _load_windows(dataset: str):
    train_d = np.load(PROCESSED / f"{dataset}_train_sequences.npz")
    val_d = np.load(PROCESSED / f"{dataset}_validation_sequences.npz")
    train_raw = load_train(dataset)
    lifetimes = train_raw.groupby("engine_id")["cycle"].max().to_dict()
    return train_d, val_d, lifetimes


def _final_cycles(engine_ids: np.ndarray, lifetimes: dict) -> np.ndarray:
    out = np.empty(len(engine_ids), dtype=int)
    i = 0
    for engine, _ in _blocks(engine_ids):
        block_len = int(np.sum(engine_ids == engine))
        out[i : i + block_len] = WINDOW + np.arange(block_len)
        i += block_len
    return out


def _blocks(engine_ids: np.ndarray):
    start = 0
    engine = engine_ids[0]
    for k in range(1, len(engine_ids)):
        if engine_ids[k] != engine:
            yield engine, k - start
            start, engine = k, engine_ids[k]
    yield engine, len(engine_ids) - start


def main() -> None:
    parser = argparse.ArgumentParser(description="RUL baseline runner")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--loss", default="mse", help="deep models: mse | huber | mae")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--notes", default="window-level engineered features")
    args = parser.parse_args()

    train_d, val_d, lifetimes = _load_windows(args.dataset)
    Xtr, ytr, ids_tr = train_d["X"], train_d["y"], train_d["engine_ids"]
    Xva, yva, ids_va = val_d["X"], val_d["y"], val_d["engine_ids"]

    t0 = time.monotonic()
    notes = args.notes
    if args.model in ("lstm", "gru", "tcn"):
        from rul_prediction.models.gru import gru_model
        from rul_prediction.models.lstm import lstm_model
        from rul_prediction.models.tcn import tcn_model
        from rul_prediction.training.callbacks import build_callbacks
        from rul_prediction.training.trainer import set_seed, train_sequence_model

        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        set_seed(args.seed)
        builder = {"lstm": lstm_model, "gru": gru_model, "tcn": tcn_model}[args.model]
        model = builder(Xtr.shape[1], Xtr.shape[2], loss=args.loss,
                        learning_rate=args.learning_rate, seed=args.seed)
        param_count = model.count_params()
        checkpoint = Path("models/checkpoints") / f"{args.dataset}_{args.model}_seed{args.seed}.weights.h5"
        history = train_sequence_model(
            model, Xtr, ytr, Xva, yva,
            batch_size=args.batch_size, epochs=args.epochs,
            callbacks=build_callbacks(checkpoint, patience=args.patience),
        )
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        notes = (f"loss={args.loss} param_count={param_count} best_epoch={best_epoch} "
                 f"version=keras3 (+ early stopping, LR reduce, checkpoint)")
        feature_desc = f"sequences:{Xtr.shape[1]}x{Xtr.shape[2]}"
    else:
        Ftr, names = extract_features(Xtr, _final_cycles(ids_tr, lifetimes))
        Fva, _ = extract_features(Xva, _final_cycles(ids_va, lifetimes))
        feature_desc = f"engineered:{len(names)}"
        if args.model == "mean":
            model = MeanBaseline().fit(Ftr, ytr)
        elif args.model == "xgboost":
            model = xgboost_regressor(args.seed)
            model.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)
        else:
            model = {"linear": linear_regressor, "rf": random_forest}[args.model](args.seed)
            model.fit(Ftr, ytr)
    train_time = time.monotonic() - t0

    if args.model in ("lstm", "gru", "tcn"):
        pred = np.clip(np.asarray(model.predict(Xva, verbose=0)).ravel(), 0, MAX_RUL)
        metrics = {
            "rmse": rmse(yva, pred),
            "mae": mae(yva, pred),
            "r2": r2(yva, pred),
            "nasa": nasa_score(yva, pred),
        }
    else:
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
        "notes": notes,
    }

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"[{args.model}] features={feature_desc}  train_time={train_time:.2f}s")
    print(f"  RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  R2={metrics['r2']:.4f}  NASA={metrics['nasa']:.3f}")
    print(f"  logged -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()