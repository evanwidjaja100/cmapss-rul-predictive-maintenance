"""Tests for the V2.1 serving core (Methodology V2.1).

The serving path (make_predictor + dev-engine scaler + shared window builder)
must reproduce the V2.1 freeze predictions exactly, and the engine-cluster
conformal interval + history flags must match the V2.1 calibration/analysis.

Terminology: observed cycles are an observed history length, never a lifetime;
there is no OOD classification - only the objective `history_is_padded` flag
and the empirical `short_history_risk_flag`.

Requires gitignored artifacts (models/, data/raw, reports/tables) — excluded
from CI via the ``needs_artifacts`` marker; run in the full local suite.
"""

import numpy as np
import pytest

from rul_prediction.data.loader import load_test
from rul_prediction.serving.v2_predictor import (
    RISK_OBSERVED_CYCLES,
    V2Predictor,
    limited_history_warning,
)

pytestmark = pytest.mark.needs_artifacts


def test_predictions_match_freeze_official_test():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    assert len(table) == 100
    assert table["engine_id"].tolist() == list(range(1, 101))
    # Bit-identical path to the freeze: engine 67 -> 156.053711, engine 78 -> 184.335968
    got = table.set_index("engine_id")
    assert abs(got.loc[67, "prediction_raw_rul"] - 156.05) < 0.01
    assert abs(got.loc[78, "prediction_raw_rul"] - 184.34) < 0.01
    assert np.isfinite(table["prediction_raw_rul"]).all()


def test_engine_cluster_conformal_interval_and_flags():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    got = table.set_index("engine_id")
    assert predictor.alpha == 0.1
    assert abs(predictor.q_cycles - 70.342926) < 0.01
    assert (abs(got["interval_width_90"] - 2 * predictor.q_cycles) < 0.01).all()
    assert (abs(got["lo_90"] - (got["prediction_raw_rul"] - predictor.q_cycles)) < 0.01).all()
    # objective padded flag: observed < window (45)
    n_padded = int(got["history_is_padded"].sum())
    assert n_padded == 4
    assert (got.loc[got["history_is_padded"], "n_cycles_observed"] < 45).all()
    assert (got.loc[~got["history_is_padded"], "n_cycles_observed"] >= 45).all()
    # empirical risk flag: observed < 128 (no OOD claim)
    n_risk = int(got["short_history_risk_flag"].sum())
    assert n_risk == 44
    assert (got.loc[got["short_history_risk_flag"], "n_cycles_observed"] < RISK_OBSERVED_CYCLES).all()
    assert "ood" not in table.columns.str.lower().tolist()


def test_limited_history_warning_text():
    assert limited_history_warning(45) is None
    assert limited_history_warning(200) is None
    assert limited_history_warning(31) == (
        "Limited observed history: only 31 cycles observed; window 45 -> 14 timesteps padded.")


def test_padded_engines_flagged_in_upload_mode_semantics():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    got = table.set_index("engine_id")
    padded = got.loc[got["history_is_padded"]]
    assert len(padded) == 4
    assert (padded["short_history_risk_flag"]).all()