"""Methodology V2: split + pseudo-test manifest + manifest evaluation tests."""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.data.pseudo_test import (
    LIFECYCLE_FRACTIONS,
    build_pseudo_test_manifest,
    load_manifest,
    save_manifest,
)
from rul_prediction.data.splitting import (
    SEED,
    read_v2_split_file,
    split_engine_ids_v2,
    write_v2_split_file,
)
from rul_prediction.evaluation.manifest import evaluate_manifest

pytestmark = pytest.mark.unit

ENGINES = set(range(1, 101))  # FD001 cardinality
LIFETIMES = {e: 128 + (e % 50) for e in ENGINES}  # deterministic synthetic lifetimes


# ---- three-way engine split -------------------------------------------------

def test_v2_split_deterministic():
    a = split_engine_ids_v2(ENGINES, seed=SEED)
    b = split_engine_ids_v2(ENGINES, seed=SEED)
    assert a == b


def test_v2_split_no_overlap():
    train, validation, calibration = split_engine_ids_v2(ENGINES, seed=SEED)
    assert train.isdisjoint(validation)
    assert train.isdisjoint(calibration)
    assert validation.isdisjoint(calibration)


def test_v2_split_counts_and_union():
    train, validation, calibration = split_engine_ids_v2(ENGINES, seed=SEED)
    assert len(train) == 70
    assert len(validation) == 15
    assert len(calibration) == 15
    assert train | validation | calibration == ENGINES


def test_v2_split_write_roundtrip(tmp_path):
    path = write_v2_split_file(ENGINES, "FD001", tmp_path, seed=SEED)
    assert path.name == "FD001_v2_seed42.json"
    train, validation, calibration = read_v2_split_file(path)
    assert len(train) == 70 and len(validation) == 15 and len(calibration) == 15
    assert train | validation | calibration == ENGINES


def test_v2_split_fd004_magnitudes():
    """Phase V2-11: FD004 split magnitudes on the 249-engine cardinality."""
    engines = set(range(1, 250))
    train, validation, calibration = split_engine_ids_v2(engines, seed=SEED)
    assert len(train) == 175 and len(validation) == 37 and len(calibration) == 37
    assert train | validation | calibration == engines


def test_v2_manifest_fd004_count():
    """Phase V2-11: FD004 manifest = 5 lifecycle fractions per engine."""
    lifetimes = {e: 128 + (e % 50) for e in range(1, 250)}
    manifest = build_pseudo_test_manifest(lifetimes)
    assert len(manifest) == 249 * len(LIFECYCLE_FRACTIONS)
    assert manifest.groupby("engine_id").size().eq(len(LIFECYCLE_FRACTIONS)).all()


# ---- pseudo-test manifests --------------------------------------------------

def test_manifest_deterministic():
    a = build_pseudo_test_manifest(LIFETIMES)
    b = build_pseudo_test_manifest(LIFETIMES)
    pd.testing.assert_frame_equal(a, b)


def test_manifest_five_samples_per_engine():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    assert manifest.groupby("engine_id").size().eq(len(LIFECYCLE_FRACTIONS)).all()
    assert len(manifest) == 100 * len(LIFECYCLE_FRACTIONS)


def test_manifest_cutoff_within_valid_range():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    assert (manifest["cutoff_cycle"] >= 1).all()
    assert (manifest["cutoff_cycle"] < manifest["full_lifetime"]).all()


def test_manifest_raw_rul_definition():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    assert (manifest["true_raw_rul"] == manifest["full_lifetime"] - manifest["cutoff_cycle"]).all()
    assert (manifest["true_raw_rul"] >= 1).all()


def test_manifest_no_duplicate_engine_cutoff():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    assert not manifest.duplicated(subset=["engine_id", "cutoff_cycle"]).any()


def test_manifest_fractions_fixed_before_performance():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    assert sorted(manifest["fraction"].unique()) == sorted(LIFECYCLE_FRACTIONS)


def test_manifest_save_load_roundtrip(tmp_path):
    manifest = build_pseudo_test_manifest(LIFETIMES)
    path = save_manifest(manifest, tmp_path / "cuts.csv")
    loaded = load_manifest(path)
    pd.testing.assert_frame_equal(loaded, manifest)


# ---- manifest evaluation API ------------------------------------------------

def _trajectories() -> dict[int, pd.DataFrame]:
    out = {}
    for engine, lifetime in LIFETIMES.items():
        out[engine] = pd.DataFrame(
            {"engine_id": engine, "cycle": np.arange(1, lifetime + 1), "sensor_1": 1.0}
        )
    return out


def test_evaluate_manifest_returns_one_prediction_per_manifest_row():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    preds = evaluate_manifest(manifest, _trajectories(), lambda history, cutoff: cutoff)
    assert preds.shape == (len(manifest),)
    np.testing.assert_array_equal(preds, manifest["cutoff_cycle"].to_numpy())


def test_evaluate_manifest_no_future_observations():
    manifest = build_pseudo_test_manifest(LIFETIMES)

    def spy(history, cutoff):
        assert int(history["cycle"].max()) <= cutoff  # leak guard: no future cycles
        return 0.0

    evaluate_manifest(manifest, _trajectories(), spy)


def test_evaluate_manifest_preprocess_applied():
    manifest = build_pseudo_test_manifest(LIFETIMES)
    seen = []

    def preprocess(history):
        seen.append(len(history))
        return history

    evaluate_manifest(manifest, _trajectories(), lambda history, cutoff: 0.0, preprocess=preprocess)
    assert len(seen) == len(manifest)


def test_evaluate_manifest_matches_manifest_row_for_row():
    manifest = build_pseudo_test_manifest(LIFETIMES)

    def predict_one(history, cutoff):
        return float(history["engine_id"].iloc[0] * 1000 + cutoff)

    preds = evaluate_manifest(manifest, _trajectories(), predict_one)
    expected = manifest["engine_id"].to_numpy() * 1000 + manifest["cutoff_cycle"].to_numpy()
    np.testing.assert_array_equal(preds, expected.astype(float))