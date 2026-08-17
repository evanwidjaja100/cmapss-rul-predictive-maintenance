"""Methodology V2.2: build fixed FD001 split/outer-eval/calibration manifests.

The outer folds reuse the V2.1 engine-group manifest (seed 42, fixed before any
model comparison). V2.2 adds the deterministic inner early-stop split scheme
(58 inner-fit / 10 inner-stop, seeds 4201..4205) and writes V2.2-namespaced
pseudo-test manifests with canonical hashes.

Outputs:
    experiments/splits/fd001_v2_2_outer_fold{1..5}_cutoffs.csv   (17 engines x 5)
    experiments/splits/fd001_v2_2_calibration_cutoffs.csv        (15 engines x 5)
    experiments/v2_2/fd001_outer_split_manifest.json             (ids + hashes)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rul_prediction.data.canonical_hash import canonical_sha256_csv, canonical_sha256_json
from rul_prediction.data.pseudo_test import V2_1_LIFECYCLE_FRACTIONS, build_pseudo_test_manifest
from rul_prediction.data.v2_1_splits import read_v2_1_cv_manifest
from rul_prediction.data.loader import load_train

SPLITS_DIR = Path("experiments/splits")
OUT_DIR = Path("experiments/v2_2")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V2.2 FD001 manifests")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default=str(SPLITS_DIR))
    args = parser.parse_args()
    splits_dir = Path(args.splits_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = load_train("FD001", args.data_dir)
    lifetimes = {int(e): int(g["cycle"].max()) for e, g in frame.groupby("engine_id")}
    payload = read_v2_1_cv_manifest(splits_dir / "fd001_v2_1_group_cv_seed42.json")
    dev_ids = set(payload["development_engine_ids"])
    cal_ids = set(payload["calibration_engine_ids"])

    outer_fold_rows = {}
    for f in payload["folds"]:
        val_ids = set(f["validation_engine_ids"])
        assert len(val_ids) == 17
        manifest = build_pseudo_test_manifest(
            {e: lifetimes[e] for e in val_ids}, V2_1_LIFECYCLE_FRACTIONS)
        path = splits_dir / f"fd001_v2_2_outer_fold{f['fold']}_cutoffs.csv"
        manifest.to_csv(path, index=False)
        legacy = splits_dir / f"fd001_v2_1_fold{f['fold']}_validation_cutoffs.csv"
        if legacy.exists():
            assert (pd.read_csv(legacy, dtype={"engine_id": int}).sort_values(
                ["engine_id", "cutoff_cycle"]).reset_index(drop=True).equals(
                manifest.sort_values(["engine_id", "cutoff_cycle"]).reset_index(drop=True))), (
                f"V2.2 outer fold {f['fold']} must equal the fixed V2.1 validation manifest")
        outer_fold_rows[f["fold"]] = {
            "engine_ids": sorted(val_ids), "n_rows": len(manifest),
            "canonical_sha256": canonical_sha256_csv(manifest),
        }

    cal_manifest = build_pseudo_test_manifest(
        {e: lifetimes[e] for e in cal_ids}, V2_1_LIFECYCLE_FRACTIONS)
    cal_path = splits_dir / "fd001_v2_2_calibration_cutoffs.csv"
    cal_manifest.to_csv(cal_path, index=False)
    legacy_cal = splits_dir / "fd001_v2_1_calibration_cutoffs.csv"
    if legacy_cal.exists():
        assert (pd.read_csv(legacy_cal, dtype={"engine_id": int}).sort_values(
            ["engine_id", "cutoff_cycle"]).reset_index(drop=True).equals(
            cal_manifest.sort_values(["engine_id", "cutoff_cycle"]).reset_index(drop=True)))

    from rul_prediction.benchmark.v2_2 import INNER_FIT_SIZE, INNER_STOP_SIZE, inner_early_stop_split
    inner = {}
    for f in payload["folds"]:
        outer_train = set(f["training_engine_ids"])
        fit_ids, stop_ids = inner_early_stop_split(outer_train, f["fold"])
        inner[f["fold"]] = {
            "inner_seed": 4200 + f["fold"],
            "inner_fit_engine_ids": sorted(fit_ids),
            "inner_stop_engine_ids": sorted(stop_ids),
        }

    manifest = {
        "methodology": "v2.2",
        "dataset": "FD001",
        "seed": 42,
        "fractions": list(V2_1_LIFECYCLE_FRACTIONS),
        "inner_split_scheme": ("random.Random(4200 + fold) on sorted outer-train IDs; "
                               f"first {INNER_FIT_SIZE} = inner-fit, next {INNER_STOP_SIZE} = inner-stop"),
        "development_engine_ids": sorted(dev_ids),
        "calibration_engine_ids": sorted(cal_ids),
        "development_engine_ids_sha256": canonical_sha256_json(sorted(dev_ids)),
        "calibration_engine_ids_sha256": canonical_sha256_json(sorted(cal_ids)),
        "outer_folds": {str(f["fold"]): outer_fold_rows[f["fold"]] for f in payload["folds"]},
        "inner_splits": {str(k): v for k, v in inner.items()},
        "calibration_manifest": {
            "engine_ids": sorted(cal_ids), "n_rows": len(cal_manifest),
            "canonical_sha256": canonical_sha256_csv(cal_manifest),
        },
    }
    out = OUT_DIR / "fd001_outer_split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"outer folds: " + ", ".join(
        f"fold{k}: {v['engine_ids'][0]}..{v['engine_ids'][-1]} ({v['n_rows']} rows)" for k, v in outer_fold_rows.items()))
    print(f"calibration: {len(cal_ids)} engines, {len(cal_manifest)} rows")


if __name__ == "__main__":
    main()