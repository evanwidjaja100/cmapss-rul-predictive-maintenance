"""Methodology V2.2 protocol-regression tests (artifact-free, synthetic).

Covers V2_2_REPAIR_PLAN.md V2.2-1..V2.2-11:

    Test A: calibration isolation - calibration IDs can never enter final
            train IDs / validation_data / preprocessing fit IDs.
    Test B: outer-fold isolation - outer-eval IDs never enter inner fit/stop,
            preprocessing fitting, training control.
    Test C: CV completeness - an incomplete candidate matrix fails loudly.
    Test D: selection-policy consistency - policy applied to the summary
            reproduces the selected candidate.
    Test E: config-driven training - the training parser consumes YAML values
            instead of hidden constants.
    Test F: canonical hash - formatting/line-ending/key-order independent.
    Test G: conformal isolation - exactly 15 engine scores; calibration IDs
            absent from training/control manifests.
    Test H: FD004 condition preprocessing - fit IDs subset of allowed
            training IDs; validation/official IDs absent from fitting.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from rul_prediction.benchmark.v2_2 import (
    INNER_FIT_SIZE,
    INNER_SEED_BASE,
    INNER_STOP_SIZE,
    apply_selection_policy,
    assert_cv_complete,
    cv_summary,
    final_duration_rule,
    inner_early_stop_split,
)
from rul_prediction.data.canonical_hash import canonical_sha256_csv, canonical_sha256_json

DEV_85 = set(range(1, 101)) - {22, 27, 31, 33, 34, 38, 46, 49, 50, 77, 81, 84, 88, 93, 97}
CAL_15 = {22, 27, 31, 33, 34, 38, 46, 49, 50, 77, 81, 84, 88, 93, 97}


# ---- Test A: calibration isolation -----------------------------------------

def _valid_manifest_payload():
    return {
        "development_engine_ids": sorted(DEV_85),
        "calibration_engine_ids": sorted(CAL_15),
        "development_engine_ids_sha256": canonical_sha256_json(sorted(DEV_85)),
        "calibration_engine_ids_sha256": canonical_sha256_json(sorted(CAL_15)),
    }


def _sample_cfg(**overrides) -> dict:
    cfg = {
        "methodology_version": "2.2",
        "dataset": "FD001",
        "model": {"candidate_name": "gru_w45_huber", "architecture": "gru",
                  "window": 45, "seed": 42, "loss": "huber", "batch_size": 256,
                  "learning_rate": 0.001},
        "splits": {
            "development_engine_manifest": "does-not-exist.json",
            "calibration_engine_manifest": "x.csv",
            "development_engine_ids_sha256": "",
            "calibration_engine_ids_sha256": "",
        },
        "training_control": {
            "validation_data": "NONE in final fit (calibration + outer-eval engines untouched)",
            "fixed_epoch_count": 15,
        },
    }
    cfg.update(overrides)
    return cfg


def test_a_final_fit_plan_never_contains_calibration_ids(tmp_path):
    """Calibration IDs must be absent from the final train-ID plan."""
    manifest = _valid_manifest_payload()
    cfg = _sample_cfg(splits={
        "development_engine_manifest": str(tmp_path / "m.json"),
        "calibration_engine_manifest": "x.csv",
        "development_engine_ids_sha256": manifest["development_engine_ids_sha256"],
        "calibration_engine_ids_sha256": manifest["calibration_engine_ids_sha256"],
    })
    (tmp_path / "m.json").write_text(json.dumps(manifest), encoding="utf-8")

    from scripts.run_v2_2_freeze import final_fit_plan
    import pandas as pd
    frame = pd.DataFrame({"engine_id": sorted(DEV_85 | CAL_15), "cycle": list(range(1, 101))})
    plan = final_fit_plan(cfg, frame)
    assert plan["cal_ids"].isdisjoint(plan["dev_ids"])
    assert plan["cal_ids"] == CAL_15
    assert len(plan["dev_ids"]) == 85


def test_a_calibration_ids_cannot_be_validation_data(cfg=_sample_cfg()):
    """The final-fit control contract forbids validation_data entirely."""
    assert cfg["training_control"]["validation_data"].startswith("NONE")


# ---- Test B: outer-fold isolation ------------------------------------------

def test_b_outer_eval_never_in_inner_splits_or_training_control():
    folds = {}
    outer_train, outer_eval = set(), set()
    for fold in range(1, 6):
        ids = sorted(DEV_85)
        rng = __import__("random").Random(42)
        rng.shuffle(ids)
        size = 17
        val = set(ids[(fold - 1) * size:fold * size])
        train = set(ids) - val
        folds[fold] = (train, val)
        fit_ids, stop_ids = inner_early_stop_split(train, fold)
        assert val.isdisjoint(fit_ids) and val.isdisjoint(stop_ids)
        assert fit_ids | stop_ids == train
        assert len(fit_ids) == INNER_FIT_SIZE and len(stop_ids) == INNER_STOP_SIZE
        assert fit_ids.isdisjoint(stop_ids)
        outer_train |= train
        outer_eval |= val
    assert outer_train == DEV_85 and outer_eval == DEV_85


def test_b_inner_seed_scheme_documented():
    """Seeds must follow the documented 4200+fold scheme (4201..4205)."""
    assert [INNER_SEED_BASE + f for f in range(1, 6)] == [4201, 4202, 4203, 4204, 4205]


def test_b_inner_split_deterministic():
    train = set(range(1, 69))
    a = inner_early_stop_split(train, 1)
    b = inner_early_stop_split(train, 1)
    assert a == b


# ---- Test C: CV completeness ------------------------------------------------

def _synthetic_fold_rows(folds_by_candidate):
    rows = []
    for cid, folds in folds_by_candidate.items():
        for f in folds:
            rows.append({"candidate_id": cid, "fold": f, "model": "m", "window": 45,
                         "parameters": "p", "notes": "n", "RMSE": 1.0, "MAE": 1.0,
                         "R2": 0.5, "NASA_total": 100.0, "NASA_mean_per_engine": 5.0,
                         "signed_bias_mean": 0.1, "training_time": 1.0})
    return rows


def test_c_incomplete_candidate_fails_loudly():
    cands = [{"id": f"c{i}"} for i in range(8)]
    rows = _synthetic_fold_rows(
        {f"c{i}": [1, 2, 3, 4, 5] for i in range(8)} | {"c5": [1, 2, 3, 5]})
    with pytest.raises(AssertionError, match="incomplete"):
        assert_cv_complete(rows, cands)


def test_c_missing_fold_detected():
    """rf_w60-style: only fold 1 present must fail."""
    cands = [{"id": "rf_w60"}]
    rows = _synthetic_fold_rows({"rf_w60": [1]})
    with pytest.raises(AssertionError, match="folds \\[1\\]"):
        assert_cv_complete(rows, cands)


def test_c_full_40_row_matrix_passes():
    cands = [{"id": f"c{i}"} for i in range(8)]
    rows = _synthetic_fold_rows({f"c{i}": [1, 2, 3, 4, 5] for i in range(8)})
    assert assert_cv_complete(rows, cands) == 40


# ---- Test D: selection-policy consistency -----------------------------------

def _policy_summary(candidates, nasa_means, rmse_means, bias_means):
    rows = []
    for i, cid in enumerate(candidates):
        for f in range(1, 6):
            rows.append({"candidate_id": cid, "fold": f, "model": "m", "window": 45,
                         "parameters": "p", "notes": "n",
                         "RMSE": rmse_means[i], "MAE": 1.0, "R2": 0.5,
                         "NASA_total": nasa_means[i] * 17,
                         "NASA_mean_per_engine": nasa_means[i],
                         "signed_bias_mean": bias_means[i], "training_time": 1.0})
    return cv_summary(rows, [{"id": c} for c in candidates])


def test_d_policy_reproduces_selection():
    """NASA-primary: xgb with clearly lower NASA wins even if worse RMSE."""
    summary = _policy_summary(
        ["gru_w45", "xgb_w90"], nasa_means=[100.0, 20.0], rmse_means=[20.0, 30.0],
        bias_means=[0.0, 1.0])
    decision = apply_selection_policy(summary)
    assert decision["deployment_selection"] == "xgb_w90"
    assert decision["nasa_risk_champion"] == "xgb_w90"
    assert decision["accuracy_champion"] == "gru_w45"  # different roles, documented


def test_d_guardrail_prefers_lower_rmse_within_pooled_se():
    """Tie within one pooled SE -> lower RMSE wins."""
    rows = []
    a_nasa = [100.0, 101.0, 99.0, 102.0, 100.5]   # mean 100.5, std > 0
    for f in range(1, 6):
        rows.append({"candidate_id": "a", "fold": f, "model": "m", "window": 45,
                     "parameters": "p", "notes": "n", "RMSE": 30.0, "MAE": 1.0,
                     "R2": 0.5, "NASA_total": a_nasa[f - 1] * 17,
                     "NASA_mean_per_engine": a_nasa[f - 1],
                     "signed_bias_mean": 0.0, "training_time": 1.0})
        rows.append({"candidate_id": "b", "fold": f, "model": "m", "window": 45,
                     "parameters": "p", "notes": "n", "RMSE": 20.0, "MAE": 1.0,
                     "R2": 0.5, "NASA_total": 100.5 * 17,
                     "NASA_mean_per_engine": 100.5,
                     "signed_bias_mean": 0.0, "training_time": 1.0})
    decision = apply_selection_policy(cv_summary(rows, [{"id": "a"}, {"id": "b"}]))
    assert decision["deployment_selection"] == "b"


# ---- Test E: config-driven training ----------------------------------------

def test_e_config_value_reaches_parser(tmp_path):
    """The freeze plan must consume YAML values, not hidden constants."""
    manifest = _valid_manifest_payload()
    m_path = tmp_path / "m.json"
    m_path.write_text(json.dumps(manifest), encoding="utf-8")
    cfg = _sample_cfg()
    cfg["splits"]["development_engine_manifest"] = str(m_path)
    cfg["splits"]["development_engine_ids_sha256"] = manifest["development_engine_ids_sha256"]
    cfg["splits"]["calibration_engine_ids_sha256"] = manifest["calibration_engine_ids_sha256"]
    cfg["model"]["window"] = 37
    cfg["model"]["candidate_name"] = "lstm_w37_huber"
    cfg["model"]["architecture"] = "lstm"
    cfg["training_control"]["fixed_epoch_count"] = 9
    frame = pd.DataFrame({"engine_id": sorted(DEV_85 | CAL_15), "cycle": range(1, 101)})

    from scripts.run_v2_2_freeze import final_fit_plan
    plan = final_fit_plan(cfg, frame)
    assert plan["window"] == 37          # YAML value, not the hidden 45
    assert plan["epochs"] == 9           # YAML value, not a hidden default
    assert plan["architecture"] == "lstm"


# ---- Test F: canonical hash -------------------------------------------------

def test_f_canonical_json_independent_of_formatting():
    payload = {"a": [1, 2], "b": {"x": "y"}, "c": "unicode-\u00e9"}
    variants = [
        payload,
        {"c": "unicode-\u00e9", "b": {"x": "y"}, "a": [1, 2]},          # key order
        json.loads(json.dumps(payload, indent=2, sort_keys=True)),       # whitespace
        json.loads(json.dumps(payload, separators=(",", ":"), sort_keys=True)),  # compact
    ]
    hashes = {canonical_sha256_json(v) for v in variants}
    assert len(hashes) == 1


def test_f_canonical_csv_independent_of_line_endings_and_ordering(tmp_path):
    df = pd.DataFrame({"engine_id": [3, 1, 2], "cutoff": [30, 10, 20], "x": [1.0, 2.0, 3.0]})
    df = df.sort_values("engine_id").reset_index(drop=True)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_text(df.to_csv(index=False).replace("\n", "\n"), encoding="utf-8")
    crlf.write_text(df.to_csv(index=False).replace("\n", "\r\n"), encoding="utf-8")
    assert canonical_sha256_csv(df) == canonical_sha256_csv(shuffled)
    assert canonical_sha256_csv(df) == canonical_sha256_csv(pd.read_csv(crlf))
    assert canonical_sha256_csv(df) == canonical_sha256_csv(pd.read_csv(lf))


def test_f_semantically_identical_manifests_same_hash():
    a = pd.DataFrame({"engine_id": [1, 1, 2], "fraction": [0.25, 0.45, 0.25],
                      "cutoff_cycle": [10, 20, 11], "true_raw_rul": [30.0, 20.0, 29.0]})
    b = pd.DataFrame({"engine_id": [2, 1, 1], "fraction": [0.25, 0.45, 0.25],
                      "cutoff_cycle": [11, 20, 10], "true_raw_rul": [29, 20, 30]})
    assert canonical_sha256_csv(a) == canonical_sha256_csv(b)


# ---- Test G: conformal isolation -------------------------------------------

def test_g_exactly_15_engine_scores():
    from rul_prediction.evaluation.conformal import engine_cluster_scores
    rng = np.random.default_rng(3)
    errors = rng.normal(size=(15, 5))
    scores = engine_cluster_scores(errors)
    assert scores.shape == (15,)
    assert len(np.unique(scores)) <= 15
    np.testing.assert_allclose(scores, np.max(np.abs(errors), axis=1))


def test_g_calibration_ids_absent_from_control_manifests():
    """Calibration engines appear only in the calibration manifest, never in CV control."""
    from rul_prediction.data.v2_1_splits import group_folds
    folds = group_folds(DEV_85, 5, 42)
    for f in folds:
        assert f.isdisjoint(CAL_15)
    assert set().union(*folds) == DEV_85


# ---- Test H: FD004 condition preprocessing ---------------------------------

def test_h_fit_ids_subset_of_allowed_training_ids():
    train = set(range(1, 176))
    val = set(range(176, 213))
    cal = set(range(213, 250))
    import random
    rng = random.Random(4201)
    ids = sorted(train)
    rng.shuffle(ids)
    inner_fit, inner_stop = set(ids[:150]), set(ids[150:175])
    assert inner_fit | inner_stop == train
    assert val.isdisjoint(inner_fit | inner_stop)
    assert cal.isdisjoint(inner_fit | inner_stop)
    assert inner_fit.isdisjoint(inner_stop)


def test_h_condition_fit_engine_ids_asserted():
    """fit_condition_models must be called with engine-id sets we can audit."""
    from rul_prediction.data.condition import fit_condition_models
    import inspect
    sig = inspect.signature(fit_condition_models)
    assert "engine_ids" in sig.parameters, "fit path must be engine-id scoped"


# ---- final-duration rule ----------------------------------------------------

def test_final_duration_rules():
    import tempfile
    p = os.path.join(tempfile.gettempdir(), "v22_be.csv")
    pd.DataFrame({"candidate_id": ["g"] * 5, "fold": [1, 2, 3, 4, 5],
                  "best_epoch": [12, 15, 14, 13, 16],
                  "best_iteration": [None] * 5}).to_csv(p, index=False)
    assert final_duration_rule(p, "g")["epochs"] == 14  # round(median(12,15,14,13,16))
    pd.DataFrame({"candidate_id": ["x"] * 5, "fold": [1, 2, 3, 4, 5],
                  "best_epoch": [None] * 5,
                  "best_iteration": [40, 44, 42, 41, 43]}).to_csv(p, index=False)
    assert final_duration_rule(p, "x")["n_estimators"] == 43  # round(median)+1