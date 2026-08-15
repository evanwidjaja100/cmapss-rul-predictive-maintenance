"""CLI: validate C-MAPSS data and build leakage-safe processed dataset variants.

Variants are parameterized by (window, RUL cap, sensor set) and stored under
data/processed/<dataset>_w<W>_c<cap>_<sensors>/ so ablations never mix data.

Usage:
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001 --validate-only
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001 --window 50 --max-rul 150 --sensors varying
    .venv/Scripts/python.exe scripts/preprocess.py --dataset FD001 --max-rul none   (uncapped RUL)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rul_prediction.data.loader import load_rul, load_test, load_train
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

SEED = 42
OUT_ROOT = Path("data/processed")


def variant_name(window: int, max_rul: int | None, sensors: str) -> str:
    cap = "none" if max_rul is None else str(max_rul)
    return f"w{window}_c{cap}_{sensors}"


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


def _build(dataset: str, data_dir: Path, split_path: Path,
           window: int, max_rul: int | None, sensors: str) -> None:
    train = load_train(dataset, data_dir)

    train_engines, val_engines = read_split_file(split_path)
    assert train_engines.isdisjoint(val_engines)

    train_part = train[train["engine_id"].isin(train_engines)]
    val_part = train[train["engine_id"].isin(val_engines)]

    # Constant sensor detection on the TRAINING partition only.
    varying = [c for c in SENSOR_COLUMNS if train_part[c].std() > 1e-12]
    constant = [c for c in SENSOR_COLUMNS if c not in varying]
    features = SENSOR_COLUMNS if sensors == "all" else varying

    scaler = fit_scaler(train_part, features)  # TRAINING ENGINES ONLY

    def _windows(part, clip=True):
        part = add_rul(part, max_rul=max_rul if clip else None, clip=clip)
        part = part.copy()
        scaled = transform(part, features, scaler)
        part[features] = scaled
        X, y, ids = make_sequences(part, features, window)
        return X, y, ids, scaled

    X_train, y_train, train_ids, train_feats = _windows(train_part)
    X_val, y_val, val_ids, val_feats = _windows(val_part)
    test_feats = transform(load_test(dataset, data_dir), features, scaler)

    assert set(np.unique(train_ids)).isdisjoint(set(np.unique(val_ids)))

    variant = variant_name(window, max_rul, sensors)
    out_dir = OUT_ROOT / f"{dataset}_{variant}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = out_dir / f"{dataset}_scaler.joblib"
    save_scaler(scaler, scaler_path)
    np.savez_compressed(out_dir / f"{dataset}_train_sequences.npz",
                        X=X_train, y=y_train, engine_ids=train_ids)
    np.savez_compressed(out_dir / f"{dataset}_validation_sequences.npz",
                        X=X_val, y=y_val, engine_ids=val_ids)
    np.savez_compressed(out_dir / f"{dataset}_scaled_features.npz",
                        train=train_feats, validation=val_feats, test=test_feats)

    metadata = {
        "dataset": dataset,
        "variant": variant,
        "window": window,
        "max_rul": max_rul,
        "sensors": sensors,
        "features": features,
        "constant_sensors_removed": constant if sensors == "varying" else [],
        "scaler_fit_partition": "TRAIN ONLY",
        "n_train_engines": len(train_engines),
        "n_validation_engines": len(val_engines),
        "n_train_sequences": int(len(y_train)),
        "n_validation_sequences": int(len(y_val)),
    }
    (out_dir / f"{dataset}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Variant: {variant} -> {out_dir}")
    print(f"Train engines: {len(train_engines)}")
    print(f"Validation engines: {len(val_engines)}")
    print(f"Overlap: {len(train_engines & val_engines)}")
    print(f"Train sequences: {len(y_train)}")
    print(f"Validation sequences: {len(y_val)}")
    print(f"Sequence length: {window}")
    print(f"Input features: {len(features)}")
    print(f"Constant sensors removed (sensors={sensors}): {constant if sensors == 'varying' else 'none'}")
    print(f"Scaler fit partition: {metadata['scaler_fit_partition']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="C-MAPSS preprocessing/validation CLI")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--max-rul", default="125", help="RUL cap; 'none' = uncapped")
    parser.add_argument("--sensors", choices=["all", "varying"], default="all")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        _validate(args.dataset, Path(args.data_dir))
        print("Validation PASSED (--validate-only: no writes performed).")
        return

    max_rul = None if args.max_rul == "none" else int(args.max_rul)
    split_path = Path(args.splits_dir) / f"{args.dataset}_seed{SEED}.json"
    if not split_path.exists():
        raise SystemExit(
            f"Split file not found: {split_path}. Run: "
            "python -m rul_prediction.data.splitting --dataset %s" % args.dataset
        )
    _validate(args.dataset, Path(args.data_dir))
    _build(args.dataset, Path(args.data_dir), split_path, args.window, max_rul, args.sensors)


if __name__ == "__main__":
    main()