"""Tests for engine-level deterministic splitting."""

import json

import pytest

from rul_prediction.data.splitting import SEED, VAL_FRACTION, split_engine_ids, write_split_file

ENGINES = set(range(1, 101))  # FD001


def test_deterministic_output():
    a_train, a_val = split_engine_ids(ENGINES, seed=SEED)
    b_train, b_val = split_engine_ids(ENGINES, seed=SEED)
    assert a_train == b_train
    assert a_val == b_val


def test_no_overlap():
    train, validation = split_engine_ids(ENGINES, seed=SEED)
    assert train.isdisjoint(validation)


def test_all_engines_accounted():
    train, validation = split_engine_ids(ENGINES, seed=SEED)
    assert train | validation == ENGINES
    assert len(train) + len(validation) == len(ENGINES)


def test_approximate_ratio():
    train, validation = split_engine_ids(ENGINES, seed=SEED)
    ratio = len(validation) / (len(train) + len(validation))
    assert ratio == pytest.approx(VAL_FRACTION, abs=0.02)


def test_seed_changes_partition():
    train_42, _ = split_engine_ids(ENGINES, seed=42)
    train_7, _ = split_engine_ids(ENGINES, seed=7)
    assert train_42 == train_42  # sanity
    assert train_42 != train_7  # different partition


def test_write_split_file_roundtrip(tmp_path):
    path = write_split_file(ENGINES, "FD001", tmp_path, seed=SEED)
    payload = json.loads(path.read_text(encoding="utf-8"))
    train_ids = set(payload["train_engine_ids"])
    validation_ids = set(payload["validation_engine_ids"])
    assert path.name == "FD001_seed42.json"
    assert train_ids.isdisjoint(validation_ids)
    assert train_ids | validation_ids == ENGINES
    assert payload["seed"] == SEED