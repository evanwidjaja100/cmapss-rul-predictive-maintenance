"""Tests for dataset-integrity validation (crafted frames, no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from rul_prediction.data.loader import DATA_COLUMNS
from rul_prediction.data.validation import validate_frame, validate_rul


def _frame(n_engines: int = 3, max_cycle: int = 10) -> pd.DataFrame:
    rows = [
        [engine, cycle] + [1.0] * 24
        for engine in range(1, n_engines + 1)
        for cycle in range(1, max_cycle + 1)
    ]
    return pd.DataFrame(rows, columns=DATA_COLUMNS)


def test_clean_frame_passes():
    assert validate_frame(_frame(100), "FD001", "train").passed


def test_duplicate_records_fail():
    frame = _frame()
    dup = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    report = validate_frame(dup, "FD001", "train")
    assert not report.passed
    assert report.checks["no duplicate (engine_id, cycle) records"] is False


def test_missing_values_fail():
    frame = _frame()
    frame.loc[5, "sensor_1"] = np.nan
    report = validate_frame(frame, "FD001", "train")
    assert report.checks["no missing values"] is False


def test_infinite_values_fail():
    frame = _frame()
    frame.loc[5, "sensor_2"] = np.inf
    report = validate_frame(frame, "FD001", "train")
    assert report.checks["no infinite values"] is False


def test_unsorted_cycles_fail():
    frame = _frame()
    swapped = frame.iloc[[1, 0]]
    rest = frame.iloc[2:]
    report = validate_frame(pd.concat([swapped, rest], ignore_index=True), "FD001", "train")
    assert report.checks["cycles strictly increasing within every engine"] is False


def test_wrong_engine_count_fails():
    report = validate_frame(_frame(3), "FD001", "train")
    assert report.checks["engine count matches dataset cardinality"] is False  # 3 != 100


def test_constant_column_reported_not_failed():
    frame = _frame(100)
    frame["sensor_1"] = 0.0
    report = validate_frame(frame, "FD001", "train")
    assert "sensor_1" in report.diagnostics["constant_columns"]
    assert report.passed  # constant sensors are a known FD001 property, not an error


def test_validate_rul_length_and_dtype():
    report = validate_rul(np.ones(100, dtype=int), "FD001")
    assert report.checks["RUL file length matches number of test engines"] is True
    assert report.checks["RUL values are positive integers"] is True
    report = validate_rul(np.ones(100, dtype=float), "FD001")
    assert report.checks["RUL values are positive integers"] is False