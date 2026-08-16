"""Tests for the V2 serving core (Methodology V2, Phase V2-9).

The serving path (make_predictor + train-only scaler + shared window builder)
must reproduce the Phase V2-5 freeze predictions exactly, and the conformal
interval + OOD flag must match the Phase V2-8 / V2-6 results.
"""

import numpy as np

from rul_prediction.data.loader import load_test
from rul_prediction.serving.v2_predictor import OOD_LIFETIME, V2Predictor


def test_predictions_match_freeze_official_test():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    assert len(table) == 100
    assert table["engine_id"].tolist() == list(range(1, 101))
    # Bit-identical path to the freeze: engine 67 -> 187.640411, engine 78 -> 196.238007
    got = table.set_index("engine_id")
    assert abs(got.loc[67, "prediction_raw_rul"] - 187.64) < 0.01
    assert abs(got.loc[78, "prediction_raw_rul"] - 196.24) < 0.01
    assert np.isfinite(table["prediction_raw_rul"]).all()


def test_conformal_interval_and_ood_flag():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    got = table.set_index("engine_id")
    assert predictor.alpha == 0.1
    assert abs(predictor.q_cycles - 24.10) < 0.01
    assert (abs(got["interval_width_90"] - 2 * predictor.q_cycles) < 0.01).all()
    assert (abs(got["lo_90"] - (got["prediction_raw_rul"] - predictor.q_cycles)) < 0.01).all()
    # V2-6 finding: 44 official engines fail below the training lifetime minimum
    n_ood = int(got["ood_short_history"].sum())
    assert n_ood == 44
    assert (got.loc[got["ood_short_history"], "n_cycles"] < OOD_LIFETIME).all()
    assert (got.loc[~got["ood_short_history"], "n_cycles"] >= OOD_LIFETIME).all()


def test_short_history_padding_matches_freezes_padded_engines():
    predictor = V2Predictor()
    table = predictor.predict_frame(load_test("FD001"))
    got = table.set_index("engine_id")
    padded = got.loc[got["n_cycles"] < 45]
    assert len(padded) == 4
    assert (padded["ood_short_history"]).all()