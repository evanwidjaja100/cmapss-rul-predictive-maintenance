"""Golden-file test: the serving pipeline must reproduce the Phase 9
one-time official test evaluation exactly (same scaler, windows, features,
model). Skipped when Phase 9 artifacts are absent."""

import csv
from pathlib import Path

import numpy as np
import pytest

from rul_prediction.data.loader import load_test
from rul_prediction.serving.inference import RulPredictor

pytestmark = [pytest.mark.integration, pytest.mark.needs_artifacts]

ROOT = Path(__file__).resolve().parents[1]
PRED_CSV = (ROOT / "data" / "processed" / "FD001_w90_c45_all" / "FD001_test_predictions.csv")
MODEL = ROOT / "models" / "final" / "FD001_final_model.joblib"


@pytest.fixture(scope="module")
def predictor():
    if not (MODEL.exists() and PRED_CSV.exists()):
        pytest.skip("Phase 9 artifacts not present")
    return RulPredictor()


def test_serving_matches_phase9_test_evaluation(predictor):
    expected = {}
    with PRED_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            expected[int(row["unit_id"])] = float(row["prediction"])

    result = predictor.predict(load_test("FD001"))
    for unit, pred in zip(result["unit_id"], result["prediction"]):
        assert abs(pred - expected[unit]) <= 1e-4, (unit, pred, expected[unit])


def test_predict_short_unit_pads(predictor):
    frame = load_test("FD001")
    short_unit = frame["engine_id"].value_counts().idxmin()
    short = frame[frame["engine_id"] == short_unit]
    result = predictor.predict(short)
    assert result["unit_id"] == [int(short_unit)]
    assert result["n_cycles"] == [len(short)]
    assert result["padded_short"] == [True] if len(short) < predictor.window else [False]