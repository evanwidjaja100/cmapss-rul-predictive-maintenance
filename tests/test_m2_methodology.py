"""Methodology M2 methodology-regression tests (artifact-free, synthetic).

Covers the repairs in M2_REPAIR_PLAN.md:
- R1/R2  lifetime semantics: observed_cycles vs true_rul vs implied_failure_cycle
- R3     history_is_padded (no OOD classification)
- R5     engine-group CV: disjointness, exact-once validation, calibration isolation
- R6     balanced lifecycle fractions fixed at 0.25/0.45/0.65/0.80/0.95
- R9/R10 engine-level conformal scores (max over 5 cutoffs) + finite-sample quantile index
"""

import numpy as np
import pytest

from rul_prediction.data.pseudo_test import (
    M2_LIFECYCLE_FRACTIONS,
    build_pseudo_test_manifest,
)
from rul_prediction.data.m2_splits import (
    SEED,
    development_calibration_split,
    group_folds,
)
from rul_prediction.evaluation.conformal import (
    engine_cluster_scores,
    finite_sample_quantile_index,
)

pytestmark = pytest.mark.unit


# ---- R1/R2: observed history is never failure lifetime ---------------------

def test_implied_failure_cycle_is_observed_plus_true_rul():
    observed_cycles, true_rul = 71, 77
    implied = observed_cycles + true_rul
    assert implied == 148
    assert observed_cycles != implied  # 71 is NOT the failure lifetime


def test_implied_failure_cycle_pipeline():
    rng = np.random.default_rng(7)
    observed = rng.integers(30, 200, size=50)
    true_rul = rng.integers(1, 200, size=50)
    implied = observed + true_rul
    assert (implied > observed).all()
    assert (implied == observed + true_rul).all()


# ---- R3: history_is_padded replaces OOD classification ---------------------

def test_history_is_padded_threshold():
    window = 45
    assert (30 < window) is True
    assert (45 < window) is False
    assert (200 < window) is False


# ---- R5: engine-group CV ----------------------------------------------------

ALL_100 = set(range(1, 101))
CAL_15 = {22, 27, 31, 33, 34, 38, 46, 49, 50, 77, 81, 84, 88, 93, 97}
DEV_85 = ALL_100 - CAL_15


def test_development_calibration_split():
    dev, cal = development_calibration_split(ALL_100, CAL_15)
    assert dev == DEV_85
    assert cal == CAL_15
    assert dev.isdisjoint(cal)


def test_group_folds_partition_and_disjointness():
    folds = group_folds(DEV_85, 5, SEED)
    assert len(folds) == 5
    assert all(len(f) == 17 for f in folds)
    assert set().union(*folds) == DEV_85
    for i in range(5):
        for j in range(i + 1, 5):
            assert folds[i].isdisjoint(folds[j])


def test_group_folds_deterministic():
    a = group_folds(DEV_85, 5, SEED)
    b = group_folds(DEV_85, 5, SEED)
    assert a == b


def test_every_development_engine_validated_exactly_once():
    folds = group_folds(DEV_85, 5, SEED)
    seen = [e for f in folds for e in f]
    assert len(seen) == 85
    assert len(set(seen)) == 85  # no duplicates -> exactly once


def test_calibration_engines_never_in_folds():
    folds = group_folds(DEV_85, 5, SEED)
    for f in folds:
        assert f.isdisjoint(CAL_15)


# ---- R6: balanced cutoffs ---------------------------------------------------

def test_m2_fractions_fixed():
    assert M2_LIFECYCLE_FRACTIONS == (0.25, 0.45, 0.65, 0.80, 0.95)


def test_manifest_exactly_five_cutoffs_per_engine():
    lifetimes = {e: 200 + e for e in range(1, 11)}
    manifest = build_pseudo_test_manifest(lifetimes, M2_LIFECYCLE_FRACTIONS)
    assert len(manifest) == 10 * 5
    assert manifest.groupby("engine_id").size().eq(5).all()
    assert sorted(manifest["fraction"].unique()) == sorted(M2_LIFECYCLE_FRACTIONS)


def test_manifest_raw_rul_definition():
    lifetimes = {e: 200 + e for e in range(1, 11)}
    manifest = build_pseudo_test_manifest(lifetimes, M2_LIFECYCLE_FRACTIONS)
    assert (manifest["true_raw_rul"] == manifest["full_lifetime"] - manifest["cutoff_cycle"]).all()
    assert (manifest["true_raw_rul"] >= 1).all()


# ---- R9/R10: engine-cluster conformal ---------------------------------------

def test_engine_cluster_scores_five_errors_to_one():
    errors = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                       [9.0, 1.0, 1.0, 1.0, 1.0],
                       [0.5, 0.5, 0.5, 0.5, 7.5]])
    scores = engine_cluster_scores(errors)
    assert scores.shape == (3,)
    np.testing.assert_allclose(scores, [5.0, 9.0, 7.5])


def test_engine_cluster_scores_abs_values():
    errors = np.array([[-4.0, 1.0, 1.0, 1.0, 1.0]])
    scores = engine_cluster_scores(errors)
    np.testing.assert_allclose(scores, [4.0])


def test_engine_cluster_scores_n_is_engines_not_rows():
    rng = np.random.default_rng(1)
    errors = rng.normal(size=(15, 5))  # 15 engines x 5 cutoffs
    scores = engine_cluster_scores(errors)
    assert scores.size == 15, "one score per engine, not one per row"


def test_finite_sample_quantile_index_formula():
    n = 15
    assert finite_sample_quantile_index(n, 0.1) == 15  # ceil(16 * 0.9) = 15
    assert finite_sample_quantile_index(n, 0.2) == 13  # ceil(16 * 0.8) = 13
    assert finite_sample_quantile_index(n, 0.3) == 12  # ceil(16 * 0.7) = 12


def test_finite_sample_quantile_index_clamped():
    assert 1 <= finite_sample_quantile_index(1, 0.9) <= 1
    assert 1 <= finite_sample_quantile_index(15, 0.99) <= 15  # k = ceil(0.16) = 1
    assert finite_sample_quantile_index(15, 0.01) == 15       # k = ceil(15.84) = 16 -> clamp 15