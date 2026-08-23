"""Methodology M2 development/calibration separation and engine-group CV.

M2 repairs (see M2_REPAIR_PLAN.md, R5/R6):

* the 15 M1 calibration engines are preserved and remain unused for model
  selection (conformal calibration only);
* the remaining 85 development engines are split into 5 engine-level folds
  (seed 42); every development engine is validated exactly once across folds;
* each fold's validation manifest uses the balanced M2 lifecycle fractions
  (0.25/0.45/0.65/0.80/0.95), fixed before any model comparison.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from rul_prediction.data.pseudo_test import M2_LIFECYCLE_FRACTIONS

SEED = 42
N_FOLDS = 5


def development_calibration_split(engine_ids, calibration_ids) -> tuple[set[int], set[int]]:
    """Return (development_ids, calibration_ids); development = all minus calibration."""
    all_ids = {int(e) for e in engine_ids}
    cal = {int(e) for e in calibration_ids}
    assert cal.issubset(all_ids), "calibration engines must be a subset of all engines"
    dev = all_ids - cal
    assert dev.isdisjoint(cal)
    return dev, cal


def group_folds(dev_ids, n_folds: int = N_FOLDS, seed: int = SEED) -> list[set[int]]:
    """Deterministic engine-group folds: n contiguous shuffled groups, disjoint, covering dev_ids."""
    ids = sorted(int(e) for e in dev_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    assert n % n_folds == 0, f"{n} development engines must divide into {n_folds} equal folds"
    size = n // n_folds
    folds = [set(ids[i * size:(i + 1) * size]) for i in range(n_folds)]
    assert all(len(f) == size for f in folds)
    union = set().union(*folds)
    assert union == set(ids), "folds must partition the development engines"
    assert sum(len(a & b) for a in folds for b in folds if a is not b) == 0, "folds must be disjoint"
    return folds


def cv_manifest_payload(dev_ids, calibration_ids, n_folds: int = N_FOLDS,
                        seed: int = SEED) -> dict:
    """Build the full M2 CV manifest payload (deterministic, JSON-serializable)."""
    dev, cal = development_calibration_split(dev_ids, calibration_ids)
    folds = group_folds(dev, n_folds, seed)
    payload = {
        "methodology": "m2",
        "dataset": "FD001",
        "seed": seed,
        "n_folds": n_folds,
        "fractions": list(M2_LIFECYCLE_FRACTIONS),
        "development_engine_ids": sorted(dev),
        "calibration_engine_ids": sorted(cal),
        "development_sha256": hashlib.sha256(str(sorted(dev)).encode()).hexdigest(),
        "calibration_sha256": hashlib.sha256(str(sorted(cal)).encode()).hexdigest(),
        "folds": [
            {"fold": i + 1, "validation_engine_ids": sorted(folds[i]),
             "training_engine_ids": sorted(set().union(*(folds[:i] + folds[i + 1:])))}
            for i in range(n_folds)
        ],
    }
    return payload


def write_m2_cv_manifest(dev_ids, calibration_ids, out_dir: str | Path,
                           n_folds: int = N_FOLDS, seed: int = SEED) -> Path:
    """Persist fd001_m2_group_cv_seed42.json and return its path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"fd001_m2_group_cv_seed{seed}.json"
    payload = cv_manifest_payload(dev_ids, calibration_ids, n_folds, seed)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def read_m2_cv_manifest(path: str | Path) -> dict:
    """Read the M2 CV manifest; returns the parsed payload dict."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assert payload["methodology"] == "m2"
    return payload


def fold_validation_engines(payload: dict) -> list[set[int]]:
    return [set(f["validation_engine_ids"]) for f in payload["folds"]]


def fold_training_engines(payload: dict) -> list[set[int]]:
    return [set(f["training_engine_ids"]) for f in payload["folds"]]