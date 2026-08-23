"""Methodology M1 raw-RUL preprocessing pipeline.

Builds the canonical M1 processed artifacts (target_mode = raw by default):
    data/processed/m1/FD001/raw/FD001_train_sequences.npz   (padded windows + raw RUL)
    data/processed/m1/FD001/raw/FD001_scaler.joblib          (fit on 70 TRAIN engines only)
    data/processed/m1/FD001/raw/FD001_metadata.json

Secondary experiment (optional, never the M1 default):
    --target-mode capped --cap 45  ->  data/processed/m1/FD001/capped45/

Legacy V1 artifacts under data/processed/FD001_* are untouched.

Usage:
    .venv/Scripts/python.exe scripts/preprocess_m1.py --dataset FD001 [--window 30]
    .venv/Scripts/python.exe scripts/preprocess_m1.py --dataset FD001 --target-mode capped --cap 45
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rul_prediction.data.loader import load_train, sensor_columns
from rul_prediction.data.preprocessing import fit_scaler, save_scaler, transform
from rul_prediction.data.splitting import SEED, read_m1_split_file
from rul_prediction.data.m1_preprocessing import (
    SENSOR_COLUMNS,
    TARGET_MODES,
    add_target,
    build_m1_train_sequences,
    target_distribution,
)
from rul_prediction.data.windows import build_window

OUT_ROOT = Path("data/processed/m1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Methodology M1 raw-RUL preprocessing")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--target-mode", choices=TARGET_MODES, default="raw",
                        help="explicit target definition; M1 default is raw (never inferred from paths)")
    parser.add_argument("--cap", type=int, default=None, help="cap for --target-mode capped only")
    args = parser.parse_args()

    if args.target_mode == "capped" and args.cap is None:
        raise SystemExit("--cap N is required with --target-mode capped")

    split_path = Path(args.splits_dir) / f"{args.dataset}_m1_seed{SEED}.json"
    if not split_path.exists():
        raise SystemExit(
            f"M1 split file not found: {split_path}. Run: "
            ".venv/Scripts/python.exe scripts/build_m1_manifests.py --dataset %s" % args.dataset
        )

    train_frame = load_train(args.dataset, args.data_dir)
    train_ids, validation_ids, calibration_ids = read_m1_split_file(split_path)
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(calibration_ids)
    assert validation_ids.isdisjoint(calibration_ids)

    train_part = train_frame[train_frame["engine_id"].isin(train_ids)]

    scaler = fit_scaler(train_part, SENSOR_COLUMNS)  # TRAINING ENGINES ONLY

    framed = add_target(train_frame, target_mode=args.target_mode, cap=args.cap)
    train_rows = framed[framed["engine_id"].isin(train_ids)]

    X_train, y_train, train_seq_ids, n_observed, masks = build_m1_train_sequences(
        transform(train_rows, SENSOR_COLUMNS, scaler),
        train_rows["engine_id"].to_numpy(),
        train_rows["rul"].to_numpy(dtype=np.float32),
        args.window,
    )

    target_dir = "raw" if args.target_mode == "raw" else f"capped{args.cap}"
    out_dir = OUT_ROOT / args.dataset / target_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    save_scaler(scaler, out_dir / f"{args.dataset}_scaler.joblib")
    np.savez_compressed(
        out_dir / f"{args.dataset}_train_sequences.npz",
        X=X_train, y=y_train, engine_ids=train_seq_ids,
        n_observed=n_observed, masks=masks,
    )

    dist = target_distribution(y_train)
    n_padded = int((n_observed < args.window).sum())
    metadata = {
        "dataset": args.dataset,
        "methodology": "m1",
        "target_mode": args.target_mode,
        "cap": args.cap if args.target_mode == "capped" else None,
        "window": args.window,
        "seed": SEED,
        "split_file": str(split_path),
        "n_train_engines": len(train_ids),
        "n_validation_engines": len(validation_ids),
        "n_calibration_engines": len(calibration_ids),
        "scaler_fit_partition": "TRAIN ENGINES ONLY",
        "n_train_sequences": int(len(y_train)),
        "n_padded_sequences": n_padded,
        "target_distribution": dist,
        "history_builder": "rul_prediction.data.windows.build_window (shared train/inference)",
    }
    (out_dir / f"{args.dataset}_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(f"M1 artifacts -> {out_dir}")
    print(f"Target mode: {args.target_mode}" + (f" (cap {args.cap})" if args.target_mode == "capped" else ""))
    print(f"Window: {args.window}")
    print(f"Training engines: {len(train_ids)} | Validation: {len(validation_ids)} | Calibration: {len(calibration_ids)}")
    print(f"Partition overlap: {len(train_ids & validation_ids) + len(train_ids & calibration_ids) + len(validation_ids & calibration_ids)}")
    print(f"Scaler fit partition: {metadata['scaler_fit_partition']}")
    print(f"Train sequences: {len(y_train)} (padded/short-history: {n_padded})")
    print(f"Raw-RUL distribution (train targets): min={dist['min']} max={dist['max']} "
          f"mean={dist['mean']:.1f} median={dist['median']:.1f} | >45: {dist['n_above_45']} | ==0: {dist['n_zero']}")


if __name__ == "__main__":
    main()