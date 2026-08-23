"""Methodology M1 feature engineering tests (artifact-free)."""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.features.m1_features import FEATURE_GROUPS, N_EXTRA_FEATURES, extract_m1_features

pytestmark = pytest.mark.unit


def _observed(n: int, n_features: int = 3, base: float = 0.0) -> np.ndarray:
    rng = np.random.RandomState(42)
    return base + rng.rand(n, n_features)


def test_feature_count_and_names():
    f, names = extract_m1_features(_observed(30), cutoff_cycle=60)
    assert f.shape == (3 * len(FEATURE_GROUPS) + N_EXTRA_FEATURES,)
    assert len(names) == len(f)
    assert names[-2] == "engine_age" and names[-1] == "history_length"
    assert names[0] == "sensor_1_latest"


def test_no_future_or_full_history_dependence():
    """Features depend only on observed rows - trailing data can never leak in."""
    full = _observed(50)
    f_full, _ = extract_m1_features(full[:30], cutoff_cycle=30)
    f_obs, _ = extract_m1_features(full[:30].copy(), cutoff_cycle=30)
    np.testing.assert_array_equal(f_full, f_obs)


def test_short_history_no_nan_and_fallbacks():
    for n in (1, 2, 4, 5, 9, 10, 11):
        f, _ = extract_m1_features(_observed(n), cutoff_cycle=7)
        assert not np.isnan(f).any()
        assert f[-1] == n  # history_length


def test_engine_age_is_cutoff_cycle():
    f1, _ = extract_m1_features(_observed(30), cutoff_cycle=60)
    f2, _ = extract_m1_features(_observed(30), cutoff_cycle=200)
    assert f1[-2] == 60.0 and f2[-2] == 200.0


def test_deterministic():
    a, _ = extract_m1_features(_observed(30), cutoff_cycle=60)
    b, _ = extract_m1_features(_observed(30), cutoff_cycle=60)
    np.testing.assert_array_equal(a, b)


def test_single_row_history():
    f, names = extract_m1_features(np.ones((1, 3)), cutoff_cycle=1)
    assert f[-1] == 1.0
    assert not np.isnan(f).any()  # slope defined as 0 for single row


# ---- classical pipeline integration (synthetic, no data artifacts) ----------

def test_classical_manifest_pipeline():
    """Linear model trained on synthetic data evaluates exactly the manifest rows."""
    from rul_prediction.data.pseudo_test import build_pseudo_test_manifest
    from sklearn.linear_model import LinearRegression

    lifetimes = {1: 150, 2: 160, 3: 170}
    manifest = build_pseudo_test_manifest(lifetimes)
    assert len(manifest) == 15

    trajectories = {}
    for engine, lifetime in lifetimes.items():
        cycles = np.arange(1, lifetime + 1, dtype=float)
        trajectories[engine] = pd.DataFrame({
            "engine_id": engine, "cycle": np.arange(1, lifetime + 1),
            "sensor_1": 0.1 * cycles + 1.0, "sensor_2": -0.05 * cycles,
        })

    # training data: one sequence per end-cycle per engine (same builder contract)
    feats, targets, names = [], [], None
    for engine, lifetime in lifetimes.items():
        for cutoff in range(1, lifetime + 1):
            history = trajectories[engine].iloc[:cutoff]
            f, names = extract_m1_features(
                history[["sensor_1", "sensor_2"]].to_numpy(), cutoff)
            feats.append(f)
            targets.append(lifetime - cutoff)
    model = LinearRegression().fit(np.stack(feats), np.array(targets))

    preds = evaluate_manifest(
        manifest, trajectories,
        lambda history, cutoff: float(
            model.predict(extract_m1_features(
                history[["sensor_1", "sensor_2"]].to_numpy(), cutoff)[0][None])[0]))

    assert preds.shape == (15,)
    assert len(preds) == len(manifest)
    assert np.all(np.isfinite(preds))