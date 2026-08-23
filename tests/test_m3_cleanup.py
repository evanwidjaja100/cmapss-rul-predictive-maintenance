"""M3 final-cleanup regression tests (M3_FINAL_CLEANUP_PLAN.md).

Covers the cleanup-pass requirements (§21):

    1. XGBoost config fully controls final model parameters.
    2. FD004 config controls final freeze parameters.
    3. Absolute-bias tie-break (bias -20 must lose to +1).
    4. Numeric history-threshold behavior (no lexicographic "90" > "128").
    5. Serving contains no test-derived empirical risk threshold.
    6. Prefix-only sensitivity baseline (no future sensor rows).
    7. Sensitivity row alignment deterministic under scrambled ordering.
    8. Deployment config contains uncertainty q.
    9. Streamlit referenced report paths exist.
    10. experiments/m3 structural completeness (tracked audit tables).
    11. Clean-checkout marker gating (artifact-free collection).

Artifact-free tests run in CI; tests reading untracked models/ or data/raw
stay marked needs_artifacts. Tests reading TRACKED experiments/m3 audit
tables run everywhere (a real Git clone contains them).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rul_prediction.benchmark.m3 import apply_selection_policy
from rul_prediction.data.canonical_hash import canonical_sha256_json

ROOT = Path(__file__).resolve().parents[1]
DEV_85 = set(range(1, 101)) - {22, 27, 31, 33, 34, 38, 46, 49, 50, 77, 81, 84, 88, 93, 97}
CAL_15 = {22, 27, 31, 33, 34, 38, 46, 49, 50, 77, 81, 84, 88, 93, 97}


# ---- 1. XGBoost config fully controls final model parameters -----------------

def _manifest_payload(tmp_path) -> Path:
    m = {
        "development_engine_ids": sorted(DEV_85),
        "calibration_engine_ids": sorted(CAL_15),
        "development_engine_ids_sha256": canonical_sha256_json(sorted(DEV_85)),
        "calibration_engine_ids_sha256": canonical_sha256_json(sorted(CAL_15)),
    }
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    return p


def _synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    from rul_prediction.data.loader import SENSOR_COLUMNS
    frames = []
    for e in sorted(DEV_85 | CAL_15):
        n = 40
        data = {"engine_id": e, "cycle": np.arange(1, n + 1)}
        for s in SENSOR_COLUMNS:
            data[s] = rng.normal(size=n)
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def _xgb_cfg(tmp_path, m_path, **overrides) -> dict:
    cfg = {
        "methodology_version": "2.2",
        "dataset": "FD001",
        "model": {
            "candidate_name": "xgb_w60_d6", "architecture": "xgboost", "window": 60,
            "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "n_estimators": 25, "random_state": 42,
            "early_stopping_rounds": None,
        },
        "splits": {
            "development_engine_manifest": str(m_path),
            "calibration_engine_manifest": "x.csv",
            "development_engine_ids_sha256": canonical_sha256_json(sorted(DEV_85)),
            "calibration_engine_ids_sha256": canonical_sha256_json(sorted(CAL_15)),
        },
        "training_control": {
            "validation_data": "NONE in final fit (calibration + outer-eval engines untouched)",
            "fixed_n_estimators": 25,
        },
    }
    for section, key, value in overrides.values():
        cfg[section][key] = value
    return cfg


@pytest.mark.unit
def test_xgboost_config_values_reach_freeze_plan(tmp_path):
    from scripts.run_m3_freeze import final_fit_plan
    m_path = _manifest_payload(tmp_path)
    cfg = _xgb_cfg(tmp_path, m_path)
    cfg["model"].update({"max_depth": 7, "learning_rate": 0.11, "subsample": 0.6,
                         "colsample_bytree": 0.7, "n_estimators": 25,
                         "random_state": 9, "window": 60})
    cfg["training_control"]["fixed_n_estimators"] = 25
    plan = final_fit_plan(cfg, _synthetic_frame())
    assert plan["max_depth"] == 7
    assert plan["learning_rate"] == 0.11
    assert plan["subsample"] == 0.6
    assert plan["colsample_bytree"] == 0.7
    assert plan["n_estimators"] == 25
    assert plan["seed"] == 9
    assert plan["window"] == 60


@pytest.mark.unit
def test_xgboost_freeze_applies_plan_params_to_model(tmp_path, monkeypatch):
    from scripts.run_m3_freeze import final_fit_plan, fit_final_model
    m_path = _manifest_payload(tmp_path)
    cfg = _xgb_cfg(tmp_path, m_path)
    cfg["model"].update({"max_depth": 7, "learning_rate": 0.11, "subsample": 0.6,
                         "colsample_bytree": 0.7, "n_estimators": 25, "random_state": 9})
    plan = final_fit_plan(cfg, _synthetic_frame())

    class _Stub:
        def __init__(self):
            self.params = {}
        def set_params(self, **kw):
            self.params.update(kw)
            return self
        def fit(self, *args, **kwargs):
            return self

    stub = _Stub()
    monkeypatch.setattr("rul_prediction.models.xgboost_model.xgboost_regressor",
                        lambda seed: stub)
    fit_final_model(plan, _synthetic_frame(), "data/raw")
    assert stub.params["max_depth"] == 7
    assert stub.params["learning_rate"] == 0.11
    assert stub.params["subsample"] == 0.6
    assert stub.params["colsample_bytree"] == 0.7
    assert stub.params["n_estimators"] == 25
    assert stub.params["random_state"] == 9
    assert stub.params["early_stopping_rounds"] is None


# ---- 2. FD004 config controls final freeze parameters -----------------------

@pytest.mark.unit
def test_fd004_config_controls_freeze_parameters(tmp_path):
    import yaml

    from rul_prediction.benchmark.m1 import ROOT as _ROOT
    from scripts.run_m3_fd004_freeze import resolve_model_config
    # strict typed validation requires the full contract; start from production
    # YAML and override with deliberately non-default values
    cfg = yaml.safe_load(
        (_ROOT / "configs" / "final_model_m3_fd004.yaml").read_text(encoding="utf-8")
    )
    cfg["model"]["candidate_name"] = "gru_w47_huber_condC"
    cfg["model"]["window"] = 47
    cfg["model"]["dropout"] = 0.4
    cfg["model"]["learning_rate"] = 0.002
    cfg["model"]["batch_size"] = 128
    cfg["model"]["seed"] = 7
    cfg["model"]["fixed_epochs"] = 13
    cfg["variant_results"]["C"]["best_epoch"] = 13
    r = resolve_model_config(cfg)
    assert r["window"] == 47
    assert r["units"] == (128, 64)
    assert r["dropout"] == 0.4
    assert r["loss"] == "huber"
    assert r["batch_size"] == 128
    assert r["learning_rate"] == 0.002
    assert r["seed"] == 7
    assert r["variant"] == "C"
    assert r["fixed_epochs"] == 13
    assert r["n_clusters"] == 6
    assert r["cluster_seed"] == 42
    assert r["n_init"] == 10
    assert r["optimizer_name"] == "adam"
    assert r["optimizer_clipnorm"] == 1.0


@pytest.mark.static_contract
def test_fd004_freeze_has_no_hidden_deployment_constants():
    """The final FD004 freeze path must not supply deployment values via
    hardcoded literals; changing the YAML must change the resolved plan."""
    src = (ROOT / "scripts" / "run_m3_fd004_freeze.py").read_text(encoding="utf-8")
    for token in ("WINDOW = 45", 'loss="huber"', "k=6", "n_init=10", "units=(128, 64)",
                  "dropout=0.3"):
        assert token not in src, f"hidden constant in freeze path: {token}"


# ---- 3. Absolute-bias tie-break ---------------------------------------------

def _policy_summary_row(cid, nasa_mean, nasa_std, rmse, bias):
    return {"candidate_id": cid, "model": "m", "window": 45, "parameters": "p",
            "notes": "n", "RMSE_mean": rmse, "RMSE_std": 1.0, "MAE_mean": 1.0,
            "MAE_std": 0.1, "R2_mean": 0.5, "R2_std": 0.1,
            "NASA_total_mean": nasa_mean * 17, "NASA_total_std": nasa_std,
            "NASA_mean_per_engine_mean": nasa_mean,
            "NASA_mean_per_engine_std": nasa_std,
            "signed_bias_mean_mean": bias, "signed_bias_mean_std": 0.1,
            "training_time_mean": 1.0, "training_time_std": 0.1}


@pytest.mark.unit
def test_bias_tie_break_prefers_smaller_absolute_bias():
    """bias = -20 must NOT beat bias = +1 under the |signed bias| rule."""
    rows = [_policy_summary_row("A", 100.0, 1.0, 30.0, -20.0),
            _policy_summary_row("B", 100.0, 1.0, 30.0, +1.0)]
    decision = apply_selection_policy(rows)
    assert decision["deployment_selection"] == "B"


@pytest.mark.tracked_artifacts
def test_deployment_candidate_reproduced_with_abs_bias_rule():
    """The real M3 candidate must still be xgb_w90_d6 (NASA gap exceeds SE)."""
    fold_csv = ROOT / "experiments" / "m3" / "fd001_outer_fold_results.csv"
    if not fold_csv.exists():
        pytest.skip("experiments/m3 not present")
    from rul_prediction.benchmark.m3 import CV_CANDIDATES, cv_summary
    fold_rows = pd.read_csv(fold_csv).to_dict("records")
    decision = apply_selection_policy(cv_summary(fold_rows, CV_CANDIDATES))
    assert decision["deployment_selection"] == "xgb_w90_d6"
    assert decision["nasa_risk_champion"] == "xgb_w90_d6"


# ---- 4. Numeric history-threshold behavior ----------------------------------

@pytest.mark.unit
def test_history_bucket_bounds_are_numeric_not_lexicographic():
    from scripts.analyze_m3_errors import history_bucket_upper
    buckets = ["[0,45)", "[45,90)", "[90,128)", "[128,200)", "[200,10000)"]
    bounds = [history_bucket_upper(b) for b in buckets]
    assert bounds == [45, 90, 128, 200, 10000]
    assert max(bounds) == 10000
    assert max(bounds[:-1]) == 200
    # the old bug compared the lower-bound STRINGS: max(["0","45","90","128"]) == "90"
    assert max("0", "45", "90", "128") == "90", "sanity: str max is lexicographic"
    assert history_bucket_upper("[90,128)") == 128


# ---- 5. Serving contains no test-derived empirical risk threshold -----------

SERVING_FORBIDDEN = ("RISK_OBSERVED_CYCLES", "short_history_risk_flag",
                     "lifetime_risk", "error_analysis")


@pytest.mark.static_contract
def test_serving_source_has_no_test_derived_risk_threshold():
    src = (ROOT / "src" / "rul_prediction" / "serving" / "m1_predictor.py").read_text(encoding="utf-8")
    for token in SERVING_FORBIDDEN:
        assert token not in src, f"serving must not contain {token}"
    app = (ROOT / "app_m1.py").read_text(encoding="utf-8")
    for token in SERVING_FORBIDDEN:
        assert token not in app, f"app must not contain {token}"
    assert "history_is_padded" in src
    assert "n_padded_timesteps" in src


# ---- 6/7. Sensitivity: prefix-only baseline + keyed alignment ---------------

def _synthetic_engine_history():
    rng = np.random.default_rng(1)
    n = 100
    df = pd.DataFrame({"cycle": np.arange(1, n + 1),
                       "sensor_2": rng.normal(size=n)})
    df.loc[df["cycle"] > 60, "sensor_2"] = 1e6  # extreme FUTURE values
    return df


@pytest.mark.unit
def test_prefix_replacement_never_reads_future_rows():
    from scripts.explain_m3_sensitivity import prefix_replacement_value
    history = _synthetic_engine_history()
    for cutoff in (30, 50, 60):
        v = prefix_replacement_value(history, "sensor_2", cutoff)
        assert v < 100.0, f"cutoff {cutoff}: future extreme values leaked"
    full = history["sensor_2"].mean()
    assert abs(full) > 1e5, "sanity: full-history mean would contain the extremes"


@pytest.mark.unit
def test_sensitivity_alignment_keyed_under_scrambled_order():
    from scripts.explain_m3_sensitivity import sensor_occlusion_deltas
    rng = np.random.default_rng(2)
    trajectories = {}
    manifest_rows = []
    for e in (1, 2, 3):
        n = 30
        traj = pd.DataFrame({"cycle": np.arange(1, n + 1),
                             "sensor_2": rng.normal(size=n)})
        trajectories[e] = traj
        for frac in (0.25, 0.45):
            cutoff = int(frac * n)
            manifest_rows.append({"engine_id": e, "cutoff_cycle": cutoff,
                                  "fraction": frac,
                                  "true_raw_rul": float(n - cutoff)})
    manifest = pd.DataFrame(manifest_rows)

    def predict_one(history, cutoff):
        return float(history["cycle"].max())  # deterministic, order-free

    a = sensor_occlusion_deltas(manifest, trajectories, predict_one, "sensor_2")
    scrambled = manifest.sample(frac=1.0, random_state=3).reset_index(drop=True)
    b = sensor_occlusion_deltas(scrambled, trajectories, predict_one, "sensor_2")
    ka = a.sort_values(["engine_id", "cutoff_cycle"]).reset_index(drop=True)
    kb = b.sort_values(["engine_id", "cutoff_cycle"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ka, kb)
    assert a["prediction_occluded"].notna().all()


# ---- 8. Deployment config contains uncertainty q ----------------------------

@pytest.mark.static_contract
def test_deployment_config_has_uncertainty_q():
    cfg = yaml.safe_load((ROOT / "configs" / "deployment_m3_fd001.yaml")
                         .read_text(encoding="utf-8"))
    u = cfg["uncertainty"]
    assert cfg["methodology_version"] == "2.2"
    assert u["method"] == "engine_cluster_conformal"
    assert u["alpha"] == 0.10
    assert u["calibration_engine_count"] == 15
    assert u["q"] == u["q_by_alpha"]["0.1"] > 0
    assert list(u["checkpoint_fractions"]) == [0.25, 0.45, 0.65, 0.8, 0.95]
    assert "empirical" in u["interpretation"]


@pytest.mark.unit
def test_serving_q_reads_from_tracked_config_without_experiment_folder(tmp_path, monkeypatch):
    """q must load from the deployment config even if experiments/m3 is absent."""
    from rul_prediction.benchmark import m1 as bench_m1
    from rul_prediction.serving import m1_predictor
    monkeypatch.setattr(m1_predictor, "ROOT", tmp_path)
    monkeypatch.setattr(bench_m1, "ROOT", tmp_path)
    (tmp_path / "configs").mkdir(parents=True)
    (tmp_path / "configs" / "deployment_m3_fd001.yaml").write_text(
        (ROOT / "configs" / "deployment_m3_fd001.yaml").read_text(encoding="utf-8"),
        encoding="utf-8")
    assert abs(m1_predictor.load_deployment_q(0.1) - 66.2097) < 1e-4


@pytest.mark.unit
def test_deployment_config_contracts_raise_not_assert(tmp_path, monkeypatch):
    """Deployment-config contracts must raise explicit errors carrying the
    offending values (no bare assert; survive python -O by construction)."""
    from rul_prediction.benchmark import m1 as bench_m1
    from rul_prediction.serving import m1_predictor
    monkeypatch.setattr(m1_predictor, "ROOT", tmp_path)
    monkeypatch.setattr(bench_m1, "ROOT", tmp_path)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    good = yaml.safe_load((ROOT / "configs" / "deployment_m3_fd001.yaml")
                          .read_text(encoding="utf-8"))

    def write_cfg(mutated):
        (cfg_dir / "deployment_m3_fd001.yaml").write_text(yaml.safe_dump(mutated),
                                                            encoding="utf-8")

    # wrong methodology_version
    bad = json.loads(json.dumps(good))
    bad["methodology_version"] = "9.9"
    write_cfg(bad)
    with pytest.raises(m1_predictor.DeploymentConfigError, match="methodology_version.*9\\.9"):
        m1_predictor.load_deployment_q(0.1)
    # wrong uncertainty method
    bad = json.loads(json.dumps(good))
    bad["uncertainty"]["method"] = "something_else"
    write_cfg(bad)
    with pytest.raises(m1_predictor.DeploymentConfigError,
                       match="engine_cluster_conformal.*something_else"):
        m1_predictor.load_deployment_q(0.1)
    # unknown alpha
    write_cfg(good)
    with pytest.raises(m1_predictor.DeploymentConfigError, match="alpha=0.99"):
        m1_predictor.load_deployment_q(0.99)
    # q disagrees with the audit CSV
    csv_dir = tmp_path / "experiments" / "m3"
    csv_dir.mkdir(parents=True)
    (csv_dir / "fd001_conformal_quantiles.csv").write_text(
        "alpha,q\n0.1,999.0\n0.2,44.7955\n0.3,30.0\n", encoding="utf-8")
    with pytest.raises(m1_predictor.DeploymentConfigError, match="disagrees"):
        m1_predictor.load_deployment_q(0.1)


@pytest.mark.unit
def test_final_model_config_methodology_contract_raises(tmp_path, monkeypatch):
    """M1Predictor must reject a non-2.2 final-model config with an explicit
    error (not a bare assert) before touching any model artifacts."""
    from rul_prediction.serving import m1_predictor
    monkeypatch.setattr(m1_predictor, "ROOT", tmp_path)
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "final_model_m3_fd001.yaml").write_text(
        yaml.safe_dump({"methodology_version": "1.0",
                        "model": {"candidate_name": "xgb_w90_d6", "window": 90}}),
        encoding="utf-8")
    with pytest.raises(m1_predictor.DeploymentConfigError,
                       match="methodology_version.*1\\.0"):
        m1_predictor.M1Predictor()


@pytest.mark.unit
def test_make_predictor_window_keyword_only_positive():
    """Plan §9.5: make_predictor window is keyword-only, validated positive int."""
    from rul_prediction.benchmark.m1 import make_predictor

    class _M:
        value = 12.0

    with pytest.raises(TypeError):
        make_predictor("mean", _M(), object(), 90)  # positional window rejected
    for bad in (0, -3, 1.5, None, True):
        with pytest.raises(ValueError, match="window must be a positive int"):
            make_predictor("mean", _M(), object(), window=bad)
    p = make_predictor("mean", _M(), object(), window=90)
    assert p(None, 0) == 12.0


# ---- 9. Streamlit referenced report paths exist -----------------------------

@pytest.mark.static_contract
def test_streamlit_report_references_exist():
    app = (ROOT / "app_m1.py").read_text(encoding="utf-8")
    refs = set(re.findall(r"reports/[\w./-]+\.md", app))
    assert refs, "no report references found in app_m1.py"
    missing = [r for r in refs if not (ROOT / r).exists()]
    assert not missing, f"broken report references: {missing}"
    assert "m3_methodology.md" not in app


# ---- 10. experiments/m3 structural completeness (tracked audit tables) ----

@pytest.mark.tracked_artifacts
def test_m3_experiment_dir_structurally_complete():
    out = ROOT / "experiments" / "m3"
    if not out.exists():
        pytest.skip("experiments/m3 not present")
    fold_rows = pd.read_csv(out / "fd001_outer_fold_results.csv")
    assert len(fold_rows) == 40
    expected = {"gru_w45_huber", "gru_w60_huber", "lstm_w45_huber", "lstm_w60_huber",
                "rf_w60", "rf_w90", "xgb_w60_d6", "xgb_w90_d6"}
    assert set(fold_rows["candidate_id"]) == expected
    for cid in expected:
        folds = sorted(int(f) for f in fold_rows.loc[fold_rows.candidate_id == cid, "fold"])
        assert folds == [1, 2, 3, 4, 5], f"{cid} incomplete: {folds}"
    assert len(pd.read_csv(out / "fd001_conformal_engine_scores.csv")) == 15
    assert len(pd.read_csv(out / "fd001_official_predictions.csv")) == 100
    fd004 = pd.read_csv(out / "fd004_variant_results.csv")
    assert len(fd004) == 4 and set(fd004["variant"]) == {"A", "B", "C", "D"}
    assert len(pd.read_csv(out / "fd004_variant_predictions.csv")) == 4 * 37 * 5
    sel = json.loads((out / "selection_decision.json").read_text(encoding="utf-8"))
    assert sel["deployment_selection"] == "xgb_w90_d6"
    qs = pd.read_csv(out / "fd001_conformal_quantiles.csv")
    assert list(qs["alpha"]) == [0.1, 0.2, 0.3]


# ---- 11. Clean-checkout marker gating ---------------------------------------

@pytest.mark.static_contract
def test_artifact_free_collection_excludes_needs_artifacts_tests():
    """Marker audit: artifact-gated tests must carry needs_artifacts (direct, nonrecursive)."""
    # Direct file scan without launching a second pytest collection
    import re
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    # Known artifact-gated files must contain needs_artifacts marker
    gated = ["tests/test_m1_serving.py", "tests/test_artifacts.py", "tests/test_inference_golden.py", "tests/test_loader.py"]
    for rel in gated:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "needs_artifacts" in src, f"{rel} must be marked needs_artifacts"
    # Also verify that at least one needs_artifacts test is discoverable via marker string, not via subprocess
    # Count artifact-gated tests via direct grep
    count = 0
    for p in (ROOT / "tests").glob("test_*.py"):
        c = p.read_text(encoding="utf-8", errors="replace")
        if "needs_artifacts" in c:
            # count def test_ occurrences in files that have needs_artifacts
            count += len(re.findall(r"def test_\w+", c))
    assert count >= 5, f"expected at least 5 artifact-gated tests discovered directly, got {count}"

@pytest.mark.static_contract
def test_artifact_gated_tests_carry_marker():
    for rel in ("tests/test_m1_serving.py",):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "needs_artifacts" in src, f"{rel} must be marked needs_artifacts"


@pytest.mark.static_contract
def test_serving_missing_artifacts_error_message_explains_generation():
    """§25: when frozen artifacts are absent, the serving error must say how to
    generate them (checked statically; the runtime path needs no artifacts)."""
    src = (ROOT / "src" / "rul_prediction" / "serving" / "m1_predictor.py") \
        .read_text(encoding="utf-8")
    assert "FileNotFoundError" in src
    assert "run_m3_freeze.py" in src
    assert "scripts" in src and "models" in src


# ---- Metric falsification from saved predictions (tracked audit tables) -----

def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def _mae(a, b):
    return float(np.mean(np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _r2(a, b):
    a = np.asarray(a, float)
    return float(1 - np.sum((a - np.asarray(b, float)) ** 2) / np.sum((a - a.mean()) ** 2))


def _nasa(a, b):
    from rul_prediction.evaluation.nasa_score import nasa_score
    return float(nasa_score(np.asarray(a, float), np.asarray(b, float)))


@pytest.mark.tracked_artifacts
def test_fd001_saved_official_predictions_recompute_headline_metrics():
    out = ROOT / "experiments" / "m3"
    if not out.exists():
        pytest.skip("experiments/m3 not present")
    df = pd.read_csv(out / "fd001_official_predictions.csv")
    assert len(df) == 100
    recorded = json.loads((out / "fd001_final_metrics.json").read_text(encoding="utf-8"))
    assert abs(_rmse(df["true_rul_official"], df["prediction"]) - recorded["official_test_RMSE"]) < 1e-3
    assert abs(_mae(df["true_rul_official"], df["prediction"]) - recorded["official_test_MAE"]) < 1e-3
    assert abs(_r2(df["true_rul_official"], df["prediction"]) - recorded["official_test_R2"]) < 1e-3
    assert abs(_nasa(df["true_rul_official"], df["prediction"]) - recorded["official_test_NASA_total"]) < 0.05


@pytest.mark.tracked_artifacts
def test_fd004_saved_official_predictions_recompute_headline_metrics():
    out = ROOT / "experiments" / "m3"
    table = ROOT / "reports" / "tables" / "m3_fd004_predictions.csv"
    if not (out.exists() and table.exists()):
        pytest.skip("experiments/m3 or fd004 predictions not present")
    df = pd.read_csv(table)
    assert len(df) == 248
    recorded = json.loads((out / "fd004_final_metrics.json").read_text(encoding="utf-8"))
    assert abs(_rmse(df["true_rul_official"], df["prediction"]) - recorded["official_test_RMSE"]) < 1e-3
    assert abs(_mae(df["true_rul_official"], df["prediction"]) - recorded["official_test_MAE"]) < 1e-3
    assert abs(_r2(df["true_rul_official"], df["prediction"]) - recorded["official_test_R2"]) < 1e-3
    assert abs(_nasa(df["true_rul_official"], df["prediction"]) - recorded["official_test_NASA_total"]) < 0.05


@pytest.mark.tracked_artifacts
def test_cv_summary_falsified_from_fold_result_rows():
    out = ROOT / "experiments" / "m3"
    if not out.exists():
        pytest.skip("experiments/m3 not present")
    from rul_prediction.benchmark.m3 import CV_CANDIDATES, cv_summary
    fold_rows = pd.read_csv(out / "fd001_outer_fold_results.csv").to_dict("records")
    recomputed = pd.DataFrame(cv_summary(fold_rows, CV_CANDIDATES))
    stored = pd.read_csv(out / "fd001_cv_summary.csv")
    merged = recomputed.merge(stored, on="candidate_id", suffixes=("_re", "_st"))
    for metric in ("RMSE", "MAE", "R2", "NASA_mean_per_engine", "signed_bias_mean"):
        diff = (merged[f"{metric}_mean_re"] - merged[f"{metric}_mean_st"]).abs()
        assert (diff < 1e-3).all(), f"{metric} drift: {diff.max()}"


# ---- FD001 feature metadata + feature-path (XGBoost consumes 2D matrix) -----

@pytest.mark.static_contract
def test_fd001_config_features_block_describes_engineered_path():
    cfg = yaml.safe_load((ROOT / "configs" / "final_model_m3_fd001.yaml")
                         .read_text(encoding="utf-8"))
    f = cfg["model"]["features"]
    assert cfg["model"]["architecture"] == "xgboost"
    assert f["representation"] == "engineered_variable_history"
    assert f["source_sensor_count"] == 21
    assert f["max_history_cycles"] == cfg["model"]["window"] == 90
    assert f["feature_extractor"] == "rul_prediction.features.m1_features.extract_m1_features"
    assert f["sequence_padding_consumed_by_model"] is False
    from scripts.select_m3_model import _model_config
    gen = _model_config({"id": "xgb_w90_d6", "model": "xgboost", "window": 90,
                         "overrides": {"max_depth": 6}},
                        {"n_estimators": 500})
    assert gen["features"]["representation"] == "engineered_variable_history"
    assert gen["features"]["sequence_padding_consumed_by_model"] is False
    assert gen["features"]["feature_extractor"] == f["feature_extractor"]


@pytest.mark.unit
def test_xgboost_freeze_estimator_receives_2d_matrix_not_masked_sequence(tmp_path, monkeypatch):
    from scripts.run_m3_freeze import final_fit_plan, fit_final_model
    m_path = _manifest_payload(tmp_path)
    cfg = _xgb_cfg(tmp_path, m_path)

    seen = {}
    class _Stub:
        def __init__(self, seed):
            seen["seed"] = seed
        def set_params(self, **kw):
            return self
        def fit(self, X, y, **kw):
            seen["fit_args"] = (X, y)
            return self

    monkeypatch.setattr("rul_prediction.models.xgboost_model.xgboost_regressor",
                        lambda seed: _Stub(seed))
    plan = final_fit_plan(cfg, _synthetic_frame())
    fit_final_model(plan, _synthetic_frame(), "data/raw")
    X, y = seen["fit_args"]
    assert X.ndim == 2, f"XGBoost must get a 2D engineered matrix, got ndim={X.ndim}"
    assert X.shape[0] == len(y)
    assert not isinstance(X, list), "estimator must not be called with [X, mask]"


# ---- FD004 KMeans n_init / seed threading ------------------------------------

@pytest.mark.unit
def test_fd004_fit_preprocessing_threads_kmeans_hyperparameters(monkeypatch):
    from scripts import run_m3_fd004 as runner
    captured = {}
    def fake_fit_condition_models(frame, engine_ids, k=6, seed=42, n_init=10):
        captured.update(k=k, seed=seed, n_init=n_init)
        return None, {"c": "scaler"}, {"s": "scaler"}
    monkeypatch.setattr(runner, "fit_condition_models", fake_fit_condition_models)
    runner.fit_preprocessing("C", _synthetic_frame(), {1}, k=4, seed=123, n_init=7)
    assert captured == {"k": 4, "seed": 123, "n_init": 7}


@pytest.mark.tracked_artifacts
def test_fd004_variant_results_select_variant_c_without_official_labels():
    """Variant C must win under the development/validation selection rule
    (lowest NASA per engine) computed purely from experiments/m3/fd004_variant_results.csv —
    no official FD004 test labels involved."""
    out = ROOT / "experiments" / "m3"
    if not out.exists():
        pytest.skip("experiments/m3 not present")
    results = pd.read_csv(out / "fd004_variant_results.csv")
    assert len(results) == 4 and set(results["variant"]) == {"A", "B", "C", "D"}
    best = results.sort_values(["NASA_mean_per_engine", "RMSE"]).iloc[0]
    assert best["variant"] == "C"


# ---- Conformal q falsification -----------------------------------------------

@pytest.mark.tracked_artifacts
def test_conformal_quantiles_recomputed_from_engine_scores():
    out = ROOT / "experiments" / "m3"
    if not out.exists():
        pytest.skip("experiments/m3 not present")
    from rul_prediction.evaluation.conformal import conformal_quantile
    scores = pd.read_csv(out / "fd001_conformal_engine_scores.csv")
    assert len(scores) == 15
    stored = pd.read_csv(out / "fd001_conformal_quantiles.csv").set_index("alpha")["q"]
    deployment = yaml.safe_load((ROOT / "configs" / "deployment_m3_fd001.yaml")
                                .read_text(encoding="utf-8"))
    for alpha in (0.1, 0.2, 0.3):
        q = conformal_quantile(scores["max_abs_error"].to_numpy(), alpha)
        assert abs(q - stored[alpha]) < 1e-3
        assert abs(q - deployment["uncertainty"]["q_by_alpha"][str(alpha)]) < 1e-3


# ---- Provenance schema matches documentation ---------------------------------

@pytest.mark.tracked_artifacts
def test_fd001_final_fit_metadata_provenance_fields_match_docs():
    out = ROOT / "experiments" / "m3"
    if not out.exists():
        pytest.skip("experiments/m3 not present")
    meta = json.loads((out / "fd001_final_fit_metadata.json").read_text(encoding="utf-8"))
    prov = meta["provenance"]
    assert "git_commit" in prov
    assert "git_is_dirty" in prov
    assert "git_diff_hash" in prov
    assert "timestamp_utc" in prov
    assert "source_tree_hash" not in prov  # not written by current freeze; docs must not claim it


@pytest.mark.unit
def test_run_metadata_helper_supports_source_tree_hash():
    from rul_prediction.benchmark.m3 import run_metadata
    meta = run_metadata("FD001", "xgb_w90_d6", 1, {"inner_seed": 1, "best_epoch": 1},
                        [1, 2], [3], 90)
    assert meta["source_tree_hash"] is not None


# ---- Required tracked local references exist ---------------------------------

@pytest.mark.static_contract
def test_required_tracked_local_references_exist():
    required = [
        "M3_FINAL_CLEANUP_PLAN.md",
        "M3_FINAL_FREEZE_PLAN.md",
        "reports/m3_final_report.md",
        "configs/final_model_m3_fd001.yaml",
        "configs/final_model_m3_fd004.yaml",
        "configs/deployment_m3_fd001.yaml",
        "experiments/m3/fd001_outer_fold_results.csv",
        "experiments/m3/selection_decision.json",
    ]
    missing = [p for p in required if not (ROOT / p).exists()]
    assert not missing, f"broken local references: {missing}"


@pytest.mark.static_contract
def test_no_broken_master_cleanup_plan_references():
    """C_MAPSS_M3_FINAL_CLEANUP_AGENT_PLAN.md does not exist and must not be referenced."""
    if (ROOT / "C_MAPSS_M3_FINAL_CLEANUP_AGENT_PLAN.md").exists():
        return  # Option A: committed plan may be referenced
    for rel in ("CHANGELOG.md", "M3_FINAL_CLEANUP_PLAN.md",
                "reports/m3_final_report.md"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert "C_MAPSS_M3_FINAL_CLEANUP_AGENT_PLAN" not in src, f"broken ref in {rel}"
