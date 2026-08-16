"""Generate Methodology V2 engine split and fixed pseudo-test manifests.

Creates (all deterministic, seed 42, never overwrites legacy split files):
    experiments/splits/fd001_v2_seed42.json           (70/15/15 engine split)
    experiments/splits/fd001_v2_validation_cutoffs.csv (75 pseudo-test rows)
    experiments/splits/fd001_v2_calibration_cutoffs.csv (75 pseudo-test rows)

Usage:
    .venv/Scripts/python.exe scripts/build_v2_manifests.py --dataset FD001
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rul_prediction.data.loader import load_train
from rul_prediction.data.pseudo_test import (
    build_pseudo_test_manifest,
    engine_lifetimes,
    manifest_summary,
    save_manifest,
)
from rul_prediction.data.splitting import (
    DEFAULT_SPLITS_DIR,
    SEED,
    split_engine_ids_v2,
    write_v2_split_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Methodology V2 split + pseudo-test manifests")
    parser.add_argument("--dataset", default="FD001")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default=str(DEFAULT_SPLITS_DIR))
    args = parser.parse_args()

    train_frame = load_train(args.dataset, args.data_dir)
    engine_ids = train_frame["engine_id"].unique()
    lifetimes = engine_lifetimes(train_frame)

    split_path = write_v2_split_file(engine_ids, args.dataset, args.out_dir, args.seed)
    out_dir = Path(args.out_dir)
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_ids = set(split["train_engine_ids"])
    validation_ids = set(split["validation_engine_ids"])
    calibration_ids = set(split["calibration_engine_ids"])
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(calibration_ids)
    assert validation_ids.isdisjoint(calibration_ids)
    assert train_ids | validation_ids | calibration_ids == set(int(e) for e in engine_ids)

    val_lifetimes = {e: lifetimes[e] for e in validation_ids}
    cal_lifetimes = {e: lifetimes[e] for e in calibration_ids}
    validation_manifest = build_pseudo_test_manifest(val_lifetimes)
    calibration_manifest = build_pseudo_test_manifest(cal_lifetimes)

    val_path = save_manifest(
        validation_manifest, out_dir / f"{args.dataset.lower()}_v2_validation_cutoffs.csv")
    cal_path = save_manifest(
        calibration_manifest, out_dir / f"{args.dataset.lower()}_v2_calibration_cutoffs.csv")

    val_summary = manifest_summary(validation_manifest)
    cal_summary = manifest_summary(calibration_manifest)

    print(f"Split file: {split_path}")
    print(f"Training engines: {split['n_train']}")
    print(f"Validation engines: {split['n_validation']}")
    print(f"Calibration engines: {split['n_calibration']}")
    print(f"Partition overlap: {len(train_ids & validation_ids) + len(train_ids & calibration_ids) + len(validation_ids & calibration_ids)}")
    print(f"Validation pseudo-test predictions: {val_summary['samples']}")
    print(f"Calibration pseudo-test predictions: {cal_summary['samples']}")
    print(f"Validation manifest -> {val_path}")
    print(f"Calibration manifest -> {cal_path}")
    print(f"Lifecycle fractions: {val_summary['fractions']}")
    print(f"Validation raw-RUL range: {val_summary['min_raw_rul']}-{val_summary['max_raw_rul']} cycles")


if __name__ == "__main__":
    main()