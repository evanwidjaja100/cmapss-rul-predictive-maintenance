"""CLI: validate C-MAPSS data and build the leakage-safe processed dataset.

Usage:
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001 --validate-only
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rul_prediction.data.loader import (
    load_rul,
    load_test,
    load_train,
)
from rul_prediction.data.preprocessing import (
    SENSOR_COLUMNS,
    add_rul,
    fit_scaler,
    save_scaler,
    transform,
)
from rul_prediction.data.sequences import make_sequences
from rul_prediction.data.splitting import read_split_file
from rul_prediction.data.validation import validate_frame, validate_rul

WINDOW = 30
MAX_RUL = 125
SEED = 42
OUT_DIR = Path("data/processed")


def _validate(dataset: str, data_dir: Path) -> None:
    train = load_train(dataset, data_dir)
    test = load_test(dataset, data_dir)
    rul = load_rul(dataset, data_dir)
    reports = [validate_frame(train, dataset, "train"), validate_frame(test, dataset, "test")]
    reports.append(validate_rul(rul, dataset))
    for report in reports:
        print("\n".join(report.lines()))
        print()
    if not all(r.passed for r in reports):
        raise SystemExit("Validation FAILED - see report above.")


def _build(dataset: str, data_dir: Path, split_path: Path) -> None:
    train = load_train(dataset, data_dir)

    train_engines, val_engines = read_split_file(split_path)
    assert train_engines.isdisjoint(val_engines)

    train_part = train[train["engine_id"].isin(train_engines)]
    val_part = train[train["engine_id"].isin(val_engines)]

    # scaler fit on TRAINING ENGINES ONLY
    scaler = fit_scaler(train_part, SENSOR_COLUMNS)

    def _windows(part, clip=True):
        part = add_rul(part, max_rul=MAX_RUL if clip else None, clip=clip)
        features = transform(part, SENSOR_COLUMNS, scaler)
        X, y, ids = make_sequences(part, SENSOR_COLUMNS, WINDOW)
        return X, y, ids, features

    X_train, y_train, train_ids, train_feats = _windows(train_part)
    X_val, y_val, val_ids, val_feats = _windows(val_part)
    test_feats = transform(load_test(dataset, data_dir), SENSOR_COLUMNS, scaler)

    assert set(np.unique(train_ids)).isdisjoint(set(np.unique(val_ids)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scaler_path = OUT_DIR / f"{dataset}_scaler.joblib"
    save_scaler(scaler, scaler_path)
    np.savez_compressed(
        OUT_DIR / f"{dataset}_train_sequences.npz", X=X_train, y=y_train, engine_ids=train_ids
    )
    np.savez_compressed(
        OUT_DIR / f"{dataset}_validation_sequences.npz", X=X_val, y=y_val, engine_ids=val_ids
    )
    np.savez_compressed(OUT_DIR / f"{dataset}_scaled_features.npz", train=train_feats,
                        validation=val_feats, test=test_feats,
                        train_engine_ids=np.array(sorted(train_engines)),
                        validation_engine_ids=np.array(sorted(val_engines)))

    metadata = {
        "dataset": dataset,
        "window": WINDOW,
        "max_rul": MAX_RUL,
        "features": SENSOR_COLUMNS,
        "scaler_fit_partition": "TRAIN ONLY",
        "scaler_path": str(scaler_path),
        "n_train_engines": len(train_engines),
        "n_validation_engines": len(val_engines),
        "n_train_sequences": int(len(y_train)),
        "n_validation_sequences": int(len(y_val)),
        "n_features": len(SENSOR_COLUMNS),
    }
    (OUT_DIR / f"{dataset}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Processed dataset written under", OUT_DIR)
    print(f"Train engines: {len(train_engines)}")
    print(f"Validation engines: {len(val_engines)}")
    print(f"Overlap: {len(train_engines & val_engines)}")
    print(f"Train sequences: {len(y_train)}")
    print(f"Validation sequences: {len(y_val)}")
    print(f"Sequence length: {WINDOW}")
    print(f"Input features: {len(SENSOR_COLUMNS)}")
    print(f"Scaler fit partition: {metadata['scaler_fit_partition']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-MAPSS preprocessing/validation CLI")
    parser.add_argument("--dataset", default="FD001", help="e.g. FD001 (default: FD001)")
    parser.add_argument("--data-dir", default="data/raw", help="raw data directory")
    parser.add_argument("--splits-dir", default="experiments/splits", help="split JSON directory")
    parser.add_argument("--validate-only", action="store_true", help="validate and summarize, write nothing")
    args = parser.parse_args()

    if args.validate_only:
        _validate(args.dataset, Path(args.data_dir))
        print("Validation PASSED (--validate-only: no writes performed).")
        return

    split_path = Path(args.splits_dir) / f"{args.dataset}_seed{SEED}.json"
    if not split_path.exists():
        raise SystemExit(
            f"Split file not found: {split_path}. Run: "
            "python -m rul_prediction.data.splitting --dataset %s" % args.dataset
        )
    _validate(args.dataset, Path(args.data_dir))
    _build(args.dataset, Path(args.data_dir), split_path)


if __name__ == "__main__":
    main()