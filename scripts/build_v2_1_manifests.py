"""Build Methodology V2.1 split and pseudo-test manifest artifacts.

Creates (all deterministic, seed 42; never overwrites V2 manifests):

FD001:
    experiments/splits/fd001_v2_1_group_cv_seed42.json       (85 dev / 15 cal + 5 folds)
    experiments/splits/fd001_v2_1_fold<1..5>_validation_cutoffs.csv  (17 engines x 5)
    experiments/splits/fd001_v2_1_calibration_cutoffs.csv    (15 engines x 5)
    experiments/splits/fd001_v2_1_calibration.json           (calibration provenance)

FD004 (keeps the V2 engine split 175/37/37):
    experiments/splits/fd004_v2_1_validation_cutoffs.csv     (37 engines x 5)
    experiments/splits/fd004_v2_1_calibration_cutoffs.csv    (37 engines x 5)
    experiments/splits/fd004_v2_1_calibration.json

Cutoffs: 0.25 / 0.45 / 0.65 / 0.80 / 0.95 (V2_1_LIFECYCLE_FRACTIONS, fixed before comparison).
The 15 FD001 calibration engine IDs are preserved from the V2 split (provenance recorded).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rul_prediction.data.loader import load_train
from rul_prediction.data.pseudo_test import (
    V2_1_LIFECYCLE_FRACTIONS,
    build_pseudo_test_manifest,
    engine_lifetimes,
    save_manifest,
)
from rul_prediction.data.splitting import DEFAULT_SPLITS_DIR, SEED, read_v2_split_file
from rul_prediction.data.v2_1_splits import write_v2_1_cv_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Methodology V2.1 splits + manifests")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default=str(DEFAULT_SPLITS_DIR))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)

    fd001_frame = load_train("FD001", args.data_dir)
    fd001_lifetimes = engine_lifetimes(fd001_frame)
    fd001_engines = set(fd001_frame["engine_id"].unique())

    v2_split_path = out_dir / "fd001_v2_seed42.json"
    assert v2_split_path.exists(), f"V2 split required for calibration provenance: {v2_split_path}"
    _, _, v2_calibration = read_v2_split_file(v2_split_path)
    v2_payload_hash = hashlib.sha256(
        json.dumps(json.loads(v2_split_path.read_text(encoding="utf-8")),
                   sort_keys=True).encode()).hexdigest()[:16]

    cv_path = write_v2_1_cv_manifest(fd001_engines, v2_calibration, out_dir)
    payload = json.loads(cv_path.read_text(encoding="utf-8"))

    for fold in payload["folds"]:
        val_ids = set(fold["validation_engine_ids"])
        manifest = build_pseudo_test_manifest(
            {e: fd001_lifetimes[e] for e in val_ids}, V2_1_LIFECYCLE_FRACTIONS)
        save_manifest(manifest, out_dir / f"fd001_v2_1_fold{fold['fold']}_validation_cutoffs.csv")
        print(f"fold {fold['fold']}: {len(val_ids)} val engines -> "
              f"fd001_v2_1_fold{fold['fold']}_validation_cutoffs.csv ({len(manifest)} rows)")

    cal_manifest = build_pseudo_test_manifest(
        {e: fd001_lifetimes[e] for e in payload["calibration_engine_ids"]},
        V2_1_LIFECYCLE_FRACTIONS)
    save_manifest(cal_manifest, out_dir / "fd001_v2_1_calibration_cutoffs.csv")
    print(f"calibration: {len(payload['calibration_engine_ids'])} engines -> "
          f"fd001_v2_1_calibration_cutoffs.csv ({len(cal_manifest)} rows)")

    cal_note = {
        "methodology": "v2.1",
        "dataset": "FD001",
        "calibration_engine_ids": payload["calibration_engine_ids"],
        "provenance": "preserved from V2 split fd001_v2_seed42.json (used only for V2-8 conformal)",
        "v2_split_sha256_prefix": v2_payload_hash,
        "development_sha256": payload["development_sha256"],
        "fractions": list(V2_1_LIFECYCLE_FRACTIONS),
    }
    (out_dir / "fd001_v2_1_calibration.json").write_text(
        json.dumps(cal_note, indent=2), encoding="utf-8")

    fd004_frame = load_train("FD004", args.data_dir)
    fd004_lifetimes = engine_lifetimes(fd004_frame)
    fd004_v2 = json.loads((out_dir / "fd004_v2_seed42.json").read_text(encoding="utf-8"))
    fd004_val = set(fd004_v2["validation_engine_ids"])
    fd004_cal = set(fd004_v2["calibration_engine_ids"])
    fd004_val_manifest = build_pseudo_test_manifest(
        {e: fd004_lifetimes[e] for e in fd004_val}, V2_1_LIFECYCLE_FRACTIONS)
    fd004_cal_manifest = build_pseudo_test_manifest(
        {e: fd004_lifetimes[e] for e in fd004_cal}, V2_1_LIFECYCLE_FRACTIONS)
    save_manifest(fd004_val_manifest, out_dir / "fd004_v2_1_validation_cutoffs.csv")
    save_manifest(fd004_cal_manifest, out_dir / "fd004_v2_1_calibration_cutoffs.csv")
    (out_dir / "fd004_v2_1_calibration.json").write_text(
        json.dumps({
            "methodology": "v2.1", "dataset": "FD004",
            "validation_engine_ids": sorted(fd004_val),
            "calibration_engine_ids": sorted(fd004_cal),
            "provenance": "preserved from V2 split fd004_v2_seed42.json",
            "fractions": list(V2_1_LIFECYCLE_FRACTIONS),
        }, indent=2), encoding="utf-8")
    print(f"FD004: {len(fd004_val)} val + {len(fd004_cal)} cal engines (V2.1 fractions)")

    print(f"CV manifest -> {cv_path}")
    print(f"Development engines: {payload['development_sha256'][:12]}... "
          f"({len(payload['development_engine_ids'])})")
    print(f"Calibration engines: {payload['calibration_sha256'][:12]}... "
          f"({len(payload['calibration_engine_ids'])})")


if __name__ == "__main__":
    main()