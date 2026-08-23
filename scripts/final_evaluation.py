"""Phase 9: freeze-final-model evaluation harness.

Runs in two stages:
  1. Reproducibility pass on VALIDATION engines (must reproduce the frozen
     config's validation metrics within tolerance, else abort before test).
  2. POST-HOC evaluation on the official NASA test set, after the config has
     been frozen and validated (labels later re-inspected in M1 audits; see
     AUDIT_M1.md Issue 7 — never described as "exactly once" in M1).

Usage:
    .venv/Scripts/python.exe scripts/final_evaluation.py --config configs/final_model.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from rul_prediction.data.loader import load_rul, load_test, load_train
from rul_prediction.data.preprocessing import load_scaler, transform
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.features.engineered_features import extract_features
from rul_prediction.models.xgboost_model import xgboost_regressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "final"


def _load_variant(dataset: str, variant: str):
    variant_dir = PROCESSED / f"{dataset}_{variant}"
    train_d = np.load(variant_dir / f"{dataset}_train_sequences.npz")
    val_d = np.load(variant_dir / f"{dataset}_validation_sequences.npz")
    scaler = load_scaler(variant_dir / f"{dataset}_scaler.joblib")
    return train_d, val_d, scaler, variant_dir


def _final_cycles(engine_ids: np.ndarray, window: int) -> np.ndarray:
    out = np.empty(len(engine_ids), dtype=int)
    i = 0
    for engine, _ in _blocks(engine_ids):
        block_len = int(np.sum(engine_ids == engine))
        out[i : i + block_len] = window + np.arange(block_len)
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


def _test_windows(test_scaled: np.ndarray, unit_ids: np.ndarray, window: int):
    """Per-unit sliding windows over scaled test rows.

    Units shorter than `window` are left-padded with zeros (scaled mean ~ 0)
    so the window always ends at the unit's last observed cycle.
    Returns (X, n_cycles, padded_flags).
    """
    Xs, n_cycles, padded = [], [], []
    for unit in np.unique(unit_ids):
        block = test_scaled[unit_ids == unit]
        n = len(block)
        if n < window:
            pad = np.zeros((window - n, block.shape[1]), dtype=np.float32)
            block = np.concatenate([pad, block])
            Xs.append(block[np.newaxis].astype(np.float32))
            n_cycles.append(n)
            padded.append(True)
        else:
            Xs.append(block[np.newaxis, n - window : n].astype(np.float32))
            n_cycles.append(n)
            padded.append(False)
    return np.concatenate(Xs, axis=0), np.array(n_cycles), np.array(padded)


def main() -> None:
    parser = argparse.ArgumentParser(description="Final model freeze + post-hoc official test evaluation")
    parser.add_argument("--config", default=str(ROOT / "configs" / "final_model.yaml"))
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--skip-validate", action="store_true",
                        help="dangerous: skip the reproducibility pass (tests only)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dataset = args.dataset
    variant = config["variant"]
    window = int(config["window"])
    max_rul = int(config["max_rul"])
    seed = int(config["seed"])

    variant_dir = PROCESSED / f"{dataset}_{variant}"
    if not variant_dir.exists():
        print("variant missing - building via preprocess.py")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "preprocess.py"), "--dataset", dataset,
             "--window", str(window), "--max-rul", str(max_rul), "--sensors", config["sensors"]],
            check=True,
        )

    train_d, val_d, scaler, variant_dir = _load_variant(dataset, variant)
    Xtr, ytr, ids_tr = train_d["X"], train_d["y"], train_d["engine_ids"]
    Xva, yva, ids_va = val_d["X"], val_d["y"], val_d["engine_ids"]

    Ftr, names = extract_features(Xtr, _final_cycles(ids_tr, window))
    Fva, _ = extract_features(Xva, _final_cycles(ids_va, window))
    print(f"features: {len(names)}  train={len(ytr)}  val={len(yva)}")

    model = xgboost_regressor(seed)
    model.fit(Ftr, ytr, eval_set=[(Fva, yva)], verbose=False)

    if not args.skip_validate:
        pred_val = np.clip(model.predict(Fva), 0, max_rul)
        val_metrics = {
            "rmse": float(rmse(yva, pred_val)),
            "mae": float(mae(yva, pred_val)),
            "r2": float(r2(yva, pred_val)),
            "nasa_score": float(nasa_score(yva, pred_val)),
        }
        print(f"validation pass: {val_metrics}")
        frozen = {k: float(config["validation_metrics"][k]) for k in val_metrics}
        ok = all(abs(val_metrics[k] - frozen[k]) <= 1e-3 for k in val_metrics)
        if not ok:
            raise SystemExit(f"REPRODUCIBILITY FAILED: got {val_metrics}, frozen {frozen} - aborting before test contact")
        print("reproducibility check PASSED against frozen config")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{dataset}_final_model.joblib"
    model.save_model(str(model_path))

    # ---- official test-set evaluation (post-hoc; see AUDIT_M1.md Issue 7) ----
    test_raw = load_test(dataset)
    features_cols = [c for c in test_raw.columns if c.startswith("sensor_")]
    test_scaled = transform(test_raw, features_cols, scaler)
    unit_ids = test_raw["engine_id"].to_numpy()
    Xte, n_cycles, padded = _test_windows(test_scaled, unit_ids, window)
    final_cycles = np.array([n for n in n_cycles])
    Fte, _ = extract_features(Xte, final_cycles)
    pred_te = np.clip(model.predict(Fte), 0, max_rul)

    true_rul = load_rul(dataset).astype(float)
    y_clip = np.minimum(true_rul, max_rul)
    metrics_clipped = {
        "rmse": float(rmse(y_clip, pred_te)),
        "mae": float(mae(y_clip, pred_te)),
        "r2": float(r2(y_clip, pred_te)),
        "nasa_score": float(nasa_score(y_clip, pred_te)),
    }
    metrics_raw = {
        "rmse": float(rmse(true_rul, pred_te)),
        "mae": float(mae(true_rul, pred_te)),
        "r2": float(r2(true_rul, pred_te)),
        "nasa_score": float(nasa_score(true_rul, pred_te)),
    }

    pred_csv = variant_dir / f"{dataset}_test_predictions.csv"
    with pred_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["unit_id", "n_cycles", "padded_short", "true_rul", "prediction"])
        for i, unit in enumerate(np.unique(unit_ids)):
            writer.writerow([int(unit), int(n_cycles[i]), bool(padded[i]),
                             int(true_rul[i]), round(float(pred_te[i]), 4)])

    result = {
        "config": config,
        "contact": "official test set, evaluated post-hoc after config freeze (first contact Phase 9; labels later re-inspected in M1 audits)",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_test_units": int(len(true_rul)),
        "n_units_short_padded": int(padded.sum()),
        "metrics_clipped_at_45": metrics_clipped,
        "metrics_raw_rul": metrics_raw,
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(pred_csv),
            "config": str(Path(args.config)),
        },
    }
    out_json = ROOT / "experiments" / f"{dataset}_final_test_results.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"TEST (clipped@{max_rul}): {metrics_clipped}")
    print(f"TEST (raw RUL)     : {metrics_raw}")
    print(f"wrote -> {out_json} and {pred_csv}")


if __name__ == "__main__":
    main()