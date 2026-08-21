"""Tests for the C-MAPSS raw-data loader (real files, skipped when absent)."""

from pathlib import Path

import pytest

from rul_prediction.data.loader import (
    DATA_COLUMNS,
    EXPECTED_ENGINE_COUNTS,
    SENSOR_COLUMNS,
    load_rul,
    load_test,
    load_train,
    sensor_columns,
    summarize,
)

RAW = Path("data/raw")
HAS_DATA = (RAW / "train_FD001.txt").exists()


@pytest.mark.integration
@pytest.mark.needs_artifacts
@pytest.mark.skipif(not HAS_DATA, reason="raw C-MAPSS files not present")
def test_train_schema_and_shape():
    frame = load_train("FD001")
    assert list(frame.columns) == DATA_COLUMNS
    assert frame.shape[1] == 26
    assert frame.shape[0] > 20000  # official FD001: 20631 rows


@pytest.mark.integration
@pytest.mark.needs_artifacts
@pytest.mark.skipif(not HAS_DATA, reason="raw C-MAPSS files not present")
def test_engine_counts():
    train, test = load_train("FD001"), load_test("FD001")
    assert train["engine_id"].nunique() == EXPECTED_ENGINE_COUNTS["FD001"]["train"]
    assert test["engine_id"].nunique() == EXPECTED_ENGINE_COUNTS["FD001"]["test"]


@pytest.mark.integration
@pytest.mark.needs_artifacts
@pytest.mark.skipif(not HAS_DATA, reason="raw C-MAPSS files not present")
def test_cycles_ordered_and_unique_per_engine():
    frame = load_train("FD001")
    grouped = frame.groupby("engine_id")["cycle"]
    assert (grouped.diff().dropna() >= 1).all()  # strictly increasing
    assert frame.duplicated(subset=["engine_id", "cycle"]).sum() == 0


@pytest.mark.integration
@pytest.mark.needs_artifacts
@pytest.mark.skipif(not HAS_DATA, reason="raw C-MAPSS files not present")
def test_rul_length():
    train = load_train("FD001")
    rul = load_rul("FD001")
    assert len(rul) == EXPECTED_ENGINE_COUNTS["FD001"]["test"] == train["engine_id"].nunique()


@pytest.mark.integration
@pytest.mark.needs_artifacts
@pytest.mark.skipif(not HAS_DATA, reason="raw C-MAPSS files not present")
def test_summary_fields():
    summary = summarize(load_train("FD001"), "FD001", "train")
    assert summary["rows"] == 20631
    assert summary["sensors"] == 21
    assert summary["operating_settings"] == 3
    assert summary["max_lifetime"] > summary["min_lifetime"]


@pytest.mark.unit
def test_sensor_columns_helper():
    import pandas as pd

    frame = pd.DataFrame(columns=DATA_COLUMNS)
    assert sensor_columns(frame) == SENSOR_COLUMNS