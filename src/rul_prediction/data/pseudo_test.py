"""Methodology V2 fixed pseudo-test manifests.

The official C-MAPSS test problem produces ONE terminal prediction per test
engine. Validation therefore uses a fixed, deterministic set of terminal
prediction points (per-engine lifecycle cutoffs) instead of scoring thousands
of overlapping sliding windows. Every model and every window setting must be
evaluated on exactly the same manifest rows, so NASA-score totals stay
comparable across experiments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Lifecycle fractions of each engine's full training lifetime, fixed BEFORE
# any performance is observed. Do not tune these against validation metrics.
LIFECYCLE_FRACTIONS: tuple[float, ...] = (0.50, 0.65, 0.80, 0.90, 0.95)

# Methodology V2.1 fractions: balanced across early / mid / late life.
# Fixed BEFORE any model comparison (V2.1 repair plan, issue R6).
V2_1_LIFECYCLE_FRACTIONS: tuple[float, ...] = (0.25, 0.45, 0.65, 0.80, 0.95)

MANIFEST_COLUMNS = ["engine_id", "full_lifetime", "cutoff_cycle", "true_raw_rul", "fraction"]


def engine_lifetimes(train_frame: pd.DataFrame) -> dict[int, int]:
    """Map engine_id -> full training lifetime in cycles (max observed cycle)."""
    return {
        int(engine): int(lifetime)
        for engine, lifetime in train_frame.groupby("engine_id")["cycle"].max().items()
    }


def build_pseudo_test_manifest(
    lifetimes: dict[int, int] | pd.Series,
    fractions: tuple[float, ...] = LIFECYCLE_FRACTIONS,
) -> pd.DataFrame:
    """Build the fixed pseudo-test manifest for a set of engines.

    One row per (engine, lifecycle fraction): the model is asked to predict
    raw RUL at that cutoff, where raw RUL = full_lifetime - cutoff_cycle.
    Deterministic: same lifetimes + fractions -> identical manifest.
    """
    lifetimes = dict(lifetimes)
    rows = []
    for engine, lifetime in sorted(lifetimes.items()):
        for fraction in fractions:
            cutoff = int(round(fraction * lifetime))
            cutoff = min(max(cutoff, 1), lifetime - 1)  # keep raw RUL in [1, lifetime-1]
            rows.append(
                {
                    "engine_id": int(engine),
                    "full_lifetime": int(lifetime),
                    "cutoff_cycle": cutoff,
                    "true_raw_rul": int(lifetime - cutoff),
                    "fraction": float(fraction),
                }
            )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    assert (manifest["cutoff_cycle"] >= 1).all()
    assert (manifest["cutoff_cycle"] < manifest["full_lifetime"]).all()
    assert (manifest["true_raw_rul"] == manifest["full_lifetime"] - manifest["cutoff_cycle"]).all()
    assert not manifest.duplicated(subset=["engine_id", "cutoff_cycle"]).any()
    return manifest


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    return path


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"engine_id": int, "cutoff_cycle": int, "full_lifetime": int})


def manifest_summary(manifest: pd.DataFrame) -> dict:
    """Diagnostic counts for acceptance reporting (never used for model selection)."""
    return {
        "samples": int(len(manifest)),
        "engines": int(manifest["engine_id"].nunique()),
        "samples_per_engine": int(manifest.groupby("engine_id").size().iloc[0]),
        "min_raw_rul": int(manifest["true_raw_rul"].min()),
        "max_raw_rul": int(manifest["true_raw_rul"].max()),
        "min_cutoff": int(manifest["cutoff_cycle"].min()),
        "max_cutoff": int(manifest["cutoff_cycle"].max()),
        "fractions": [float(f) for f in sorted(manifest["fraction"].unique())],
    }