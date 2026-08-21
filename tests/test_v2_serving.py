"""Tests for the V2.2 serving core (Methodology V2.2).

The serving path (make_predictor + dev-engine scaler + shared window builder)
must reproduce the V2.2 freeze predictions exactly, and the engine-cluster
conformal interval + objective history fields must match the V2.2
calibration.

Terminology: observed cycles are an observed history length, never a lifetime.
Serving exposes only OBJECTIVE facts: `history_is_padded` and
`n_padded_timesteps`. No OOD classification and NO empirical risk threshold
(a post-hoc official-test-derived threshold must not drive serving).

Requires gitignored artifacts (models/, data/raw, experiments/v2_2) — excluded
from CI via the ``needs_artifacts`` marker; run in the full local suite.
"""

import numpy as np
import pytest

from rul_prediction.data.loader import load_test
from rul_prediction.serving.v2_predictor import (
    V2Predictor,
    limited_history_warning,
    load_deployment_q,
)

pytestmark = [pytest.mark.integration, pytest.mark.needs_artifacts]


def test_predictions_match_freeze_official_test():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    assert len(table) == 100
    assert table["engine_id"].tolist() == list(range(1, 101))
    # Bit-identical path to the V2.2 freeze: engine 67 -> 186.80, engine 78 -> 167.19
    got = table.set_index("engine_id")
    assert abs(got.loc[67, "prediction_raw_rul"] - 186.80) < 0.02
    assert abs(got.loc[78, "prediction_raw_rul"] - 167.19) < 0.02
    assert np.isfinite(table["prediction_raw_rul"]).all()


def test_engine_cluster_conformal_interval_and_padding_fields():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    got = table.set_index("engine_id")
    assert predictor.alpha == 0.1
    assert abs(predictor.q_cycles - 66.2097) < 0.01
    assert (abs(got["interval_width_90"] - 2 * predictor.q_cycles) < 0.01).all()
    assert (abs(got["lo_90"] - (got["prediction_raw_rul"] - predictor.q_cycles)) < 0.01).all()
    assert predictor.window == 90
    assert predictor.model_version.startswith("v2.2-")
    # objective padded flag: observed < window (90)
    n_padded = int(got["history_is_padded"].sum())
    assert n_padded == 26
    assert (got.loc[got["history_is_padded"], "n_cycles_observed"] < 90).all()
    assert (got.loc[~got["history_is_padded"], "n_cycles_observed"] >= 90).all()
    assert (got.loc[got["history_is_padded"], "n_padded_timesteps"] > 0).all()
    assert (got.loc[got["history_is_padded"], "n_padded_timesteps"] ==
            90 - got.loc[got["history_is_padded"], "n_cycles_observed"]).all()
    # no OOD, no lifetime-risk, no empirical risk flag
    cols = table.columns.str.lower().tolist()
    assert "ood" not in cols
    assert "out_of_distribution" not in cols
    assert "lifetime_risk" not in cols
    assert "risk" not in cols
    assert "model_version" in table.columns
    assert "calibration_method" in table.columns
    assert "predefined lifecycle checkpoints" in predictor.calibration_method
    assert "engineering extrapolation" in predictor.uncertainty_disclosure


def test_limited_history_warning_text():
    assert limited_history_warning(90, 90) is None
    assert limited_history_warning(200, 90) is None
    assert limited_history_warning(31, 90) == (
        "Limited observed history: only 31 cycles observed; window 90 -> 59 timesteps padded.")


def test_q_comes_from_tracked_deployment_config():
    predictor = V2Predictor()
    assert abs(load_deployment_q(0.1) - 66.2097) < 1e-4
    assert abs(load_deployment_q(0.2) - 44.7955) < 1e-4
    assert predictor.q_cycles == load_deployment_q(0.1)


def test_padded_engines_flagged_in_upload_mode_semantics():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    got = table.set_index("engine_id")
    padded = got.loc[got["history_is_padded"]]
    assert len(padded) == 26