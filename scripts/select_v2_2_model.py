"""Methodology V2.2: apply the pre-specified selection policy to the CV results.

Mechanical application of the rule locked in V2_2_REPAIR_PLAN.md in the
recorded development session BEFORE the V2.2 CV comparison results were
inspected (the plan file itself was uncommitted at run time, so Git cannot
prove a formal pre-registration; the policy is pre-specified, not
pre-registered):

    PRIMARY: lowest mean NASA per engine
    GUARDRAIL: within one pooled standard error of the best -> prefer lower RMSE
    TIE: smaller |signed bias|  (implemented as abs(signed_bias_mean))

Reports accuracy champion, NASA-risk champion and deployment selection
separately, writes selection_decision.json, and generates
configs/final_model_v2_2_fd001.yaml (model-type-specific: XGBoost candidates
carry XGBoost hyperparameters; deep candidates carry deep-model fields; all
values derived from the selection + manifest + the shared model factories).

Usage: python scripts/select_v2_2_model.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from rul_prediction.benchmark.v2_2 import (
    CV_CANDIDATES,
    apply_selection_policy,
    cv_summary,
    final_duration_rule,
)
from rul_prediction.data.canonical_hash import canonical_sha256_json, canonical_sha256_csv

OUT_DIR = Path("experiments/v2_2")

XGB_PARAM_NAMES = ("max_depth", "learning_rate", "subsample", "colsample_bytree",
                   "n_estimators", "random_state", "early_stopping_rounds")


def _xgb_factory_params(seed: int) -> dict:
    """Current XGBoost factory defaults (single source: xgboost_model.py)."""
    from rul_prediction.models.xgboost_model import xgboost_regressor
    model = xgboost_regressor(seed)
    return {k: getattr(model, k) for k in XGB_PARAM_NAMES}


def _model_config(candidate: dict, duration: dict) -> dict:
    """Model-type-specific YAML block: XGBoost fields or deep-model fields."""
    arch = candidate["model"]
    base = {
        "candidate_name": candidate["id"],
        "architecture": arch,
        "window": candidate["window"],
    }
    if arch == "xgboost":
        params = _xgb_factory_params(42)
        params["max_depth"] = (candidate["overrides"] or {}).get(
            "max_depth", params["max_depth"])
        params["n_estimators"] = duration["n_estimators"]
        params["early_stopping_rounds"] = None  # final fit: fixed duration, no stopping
        base["features"] = {
            "representation": "engineered_variable_history",
            "source_sensor_count": 21,
            "max_history_cycles": candidate["window"],
            "feature_extractor": "rul_prediction.features.v2_features.extract_v2_features",
            "sequence_padding_consumed_by_model": False,
        }
        return {**base, **params}
    base["features"] = (f"{candidate['window']}-cycle windows of 21 sensors "
                        "(padded+masked, shared history builder)")
    return {**base,
        "units": [128, 64],
        "dropout": 0.3,
        "loss": (candidate["overrides"] or {}).get("loss", "mse"),
        "optimizer": "Adam(clipnorm=1.0)",
        "learning_rate": 0.001,
        "batch_size": 256,
        "seed": 42,
        "epochs": duration["epochs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply V2.2 selection policy")
    parser.add_argument("--config-out", default="configs/final_model_v2_2_fd001.yaml")
    args = parser.parse_args()

    fold_rows = pd.read_csv(OUT_DIR / "fd001_outer_fold_results.csv").to_dict("records")
    summary = cv_summary(fold_rows, CV_CANDIDATES)
    decision = apply_selection_policy(summary)
    selected = decision["deployment_selection"]

    duration = final_duration_rule(OUT_DIR / "fd001_best_epochs.csv", selected)
    candidate = next(c for c in CV_CANDIDATES if c["id"] == selected)
    decision["final_duration_rule"] = duration

    split_manifest = json.loads(
        (OUT_DIR / "fd001_outer_split_manifest.json").read_text(encoding="utf-8"))
    decision["manifest_hashes"] = {
        "development_engine_ids_sha256": split_manifest["development_engine_ids_sha256"],
        "calibration_engine_ids_sha256": split_manifest["calibration_engine_ids_sha256"],
        "outer_folds": split_manifest["outer_folds"],
        "calibration_manifest": split_manifest["calibration_manifest"],
    }

    (OUT_DIR / "selection_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))

    import importlib.metadata as md
    versions = {}
    for pkg in ("tensorflow", "numpy", "pandas", "scikit-learn", "xgboost", "joblib"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    import sys
    versions["python"] = sys.version.split()[0]

    summary_df = pd.DataFrame(summary)
    sel_row = summary_df[summary_df.candidate_id == selected].iloc[0]
    model_cfg = _model_config(candidate, duration)
    cfg = {
        "methodology_version": "2.2",
        "dataset": "FD001",
        "target": "raw RUL regression",
        "model": model_cfg,
        "preprocessing": {"mode": "StandardScaler fit on permitted training rows only"},
        "splits": {
            "development_engine_count": 85,
            "calibration_engine_count": 15,
            "outer_fold_count": 5,
            "inner_split": "58 inner-fit / 10 inner-stop, seeds 4201..4205",
            "development_engine_manifest": "experiments/v2_2/fd001_outer_split_manifest.json",
            "calibration_engine_manifest": "experiments/splits/fd001_v2_2_calibration_cutoffs.csv",
            "development_engine_ids_sha256": split_manifest["development_engine_ids_sha256"],
            "calibration_engine_ids_sha256": split_manifest["calibration_engine_ids_sha256"],
            "outer_cv_manifest": "experiments/v2_2/fd001_outer_split_manifest.json",
            "canonical_hashes": {
                "outer_fold_manifests": split_manifest["outer_folds"],
                "calibration_manifest": split_manifest["calibration_manifest"],
            },
        },
        "training_control": {
            "rule": duration["rule"],
            "best_epochs": duration.get("best_epochs"),
            "best_iterations": duration.get("best_iterations"),
            "fixed_epoch_count": duration.get("epochs"),
            "fixed_n_estimators": duration.get("n_estimators"),
            "validation_data": "NONE in final fit (calibration + outer-eval engines untouched)",
        },
        "selection_policy": decision["rule"],
        "selection": {
            "accuracy_champion": decision["accuracy_champion"],
            "nasa_risk_champion": decision["nasa_risk_champion"],
            "deployment_selection": decision["deployment_selection"],
            "pooled_se_ties": decision["pooled_se_ties"],
            "cv_results_csv": "experiments/v2_2/fd001_cv_summary.csv",
            "cv_metrics": {k: float(sel_row[k]) for k in (
                "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std", "R2_mean", "R2_std",
                "NASA_mean_per_engine_mean", "NASA_mean_per_engine_std",
                "signed_bias_mean_mean", "signed_bias_mean_std")},
        },
        "software_versions": versions,
    }

    out = Path(args.config_out)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"deployment selection: {selected}; duration rule: {duration}")


if __name__ == "__main__":
    main()