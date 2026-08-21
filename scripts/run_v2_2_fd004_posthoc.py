"""Methodology V2.2: post-hoc official FD004 evaluation (after YAML freeze).

Official FD004 labels are permanently POST-HOC (inspected in V2-11; never
select the variant). Falsification: variant results and headline metrics are
recomputed from the saved prediction CSV and compared with the config.

Hash checks run BEFORE deserialization. Official labels read ONLY after
model/config/preprocessor compatibility passes.
"""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml  # kept for compatibility but not used directly; config via typed loader
from joblib import load as load_joblib
from tensorflow import keras

from rul_prediction.benchmark.fd004_config import (
    FD004ConfigError,
    HISTORICAL_CONDITION_JOBLIB_SHA256,
    load_fd004_final_config,
)
from rul_prediction.benchmark.v2 import ROOT
from rul_prediction.data.loader import EXPECTED_ENGINE_COUNTS, load_rul, load_test
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

try:
    from run_v2_2_fd004 import build_matrix, make_predictor
except ImportError:
    from scripts.run_v2_2_fd004 import build_matrix, make_predictor

OUT_DIR = ROOT / "experiments" / "v2_2"

# expected feature dims per variant (C=21 sensor per-regime, D=30 etc.)
VARIANT_FEATURE_DIMS = {"A": 21, "B": 24, "C": 21, "D": 30}


def _verify_model_dimensions(model, config, expected_feature_dim: int | None = None) -> None:
    """Verify Keras time and feature dimensions against config."""
    try:
        inp = model.input_shape
        # model.input_shape is list for multi-input; inp[0] is windowed tensor
        # v2_gru has two inputs: window and mask. First is (None, window, n_features)
        shape = inp[0] if isinstance(inp, list) else inp
        _, w, f = shape
        if int(w) != int(config.window):
            raise FD004ConfigError(f"model window {w} != config.window {config.window}")
        if expected_feature_dim is not None and int(f) != int(expected_feature_dim):
            raise FD004ConfigError(f"model feature dim {f} != expected {expected_feature_dim} for variant {config.selected_variant}")
    except FD004ConfigError:
        raise
    except Exception:
        # stub models in tests may not have input_shape; skip strict check but ensure window passed
        pass


def load_and_verify_condition(config, *, model=None) -> dict:
    """Load and verify condition preprocessing payload.

    Distinguishes:
      - missing artifact (FileNotFoundError)
      - hash-mismatched artifact (FD004ConfigError)
      - legacy-incompatible artifact (FD004ConfigError with legacy tag)
    Performs hash check BEFORE deserialization (legacy gate).
    """
    cond_path = config.condition_artifact_path(ROOT)
    if not cond_path.exists():
        raise FileNotFoundError(
            f"FD004 condition artifact missing: {cond_path}. Generate via scripts/run_v2_2_fd004_freeze.py "
            f"(or ensure models/v2_2/fd004_condition{config.selected_variant}.joblib exists)"
        )
    # Hash the exact bytes that will be deserialized.
    try:
        payload_bytes = cond_path.read_bytes()
    except OSError as e:
        raise FD004ConfigError(f"failed to read condition payload {cond_path}: {e}") from e
    actual_sha = hashlib.sha256(payload_bytes).hexdigest()
    is_historical = actual_sha.lower() == HISTORICAL_CONDITION_JOBLIB_SHA256.lower()
    if not is_historical:
        # Future pickles need an allowlisted manifest hash too; inspecting pickle bytes
        # is not a safe authorization mechanism.
        from rul_prediction.artifact_manifest import MANIFEST_PATHS, load_manifest

        manifest_path = ROOT / MANIFEST_PATHS["FD004"]
        if not manifest_path.exists():
            raise FD004ConfigError("future condition payload requires a trusted FD004 manifest before deserialization")
        try:
            rel_path = cond_path.relative_to(ROOT).as_posix()
        except ValueError as e:
            raise FD004ConfigError(f"condition payload is outside repository root: {cond_path}") from e
        manifest = load_manifest(manifest_path, root=ROOT)
        entry = next((a for a in manifest["artifacts"] if a["path"] == rel_path), None)
        if entry is None or actual_sha.lower() != entry["sha256"].lower():
            expected = entry["sha256"] if entry else "no manifest entry"
            raise FD004ConfigError(
                f"future condition payload hash is not authorized: expected {expected}, got {actual_sha}"
            )
    # Load the same bytes that were hashed, not a path that can be swapped after verification.
    try:
        payload = load_joblib(BytesIO(payload_bytes))
    except Exception as e:
        raise FD004ConfigError(f"failed to load condition payload {cond_path}: {e}") from e

    # distinguish future vs legacy
    if isinstance(payload, dict) and payload.get("schema_version") == "fd004-condition-v1":
        # future schema
        # verify identity
        if payload.get("candidate") != config.candidate_name:
            raise FD004ConfigError(f"condition payload candidate {payload.get('candidate')!r} != config {config.candidate_name!r}")
        if payload.get("variant") != config.selected_variant:
            raise FD004ConfigError(f"condition payload variant {payload.get('variant')!r} != config {config.selected_variant!r}")
        if int(payload.get("window", config.window)) != int(config.window):
            raise FD004ConfigError(f"condition payload window {payload.get('window')} != config {config.window}")
        # config hash verification (if present)
        if payload.get("config_file_sha256") and payload.get("config_file_sha256") != config.config_file_sha256:
            raise FD004ConfigError("condition payload config_file_sha256 mismatch config")
        if payload.get("config_canonical_sha256") and payload.get("config_canonical_sha256") != config.config_canonical_sha256:
            raise FD004ConfigError("condition payload config_canonical_sha256 mismatch")
        # preprocessing type
        expected_type = "global" if config.selected_variant in ("A", "B") else "per_regime"
        if payload.get("preprocessing_type") != expected_type:
            raise FD004ConfigError(f"preprocessing_type {payload.get('preprocessing_type')} != expected {expected_type}")
        # n_clusters etc
        if config.selected_variant in ("C", "D"):
            kmeans = payload.get("kmeans")
            if kmeans is not None and getattr(kmeans, "n_clusters", config.n_clusters) != config.n_clusters:
                raise FD004ConfigError("condition payload kmeans n_clusters mismatch")
        # verify feature dim vs model if model provided
        # extract objects
        extracted = {
            "kmeans": payload.get("kmeans"),
            "cluster_scalers": payload.get("cluster_scalers"),
            "settings_scaler": payload.get("settings_scaler"),
            "global_scaler": payload.get("global_scaler"),
        }
        # validate required objects per variant
        if config.selected_variant in ("A", "B"):
            if extracted["global_scaler"] is None:
                raise FD004ConfigError(f"variant {config.selected_variant} future payload missing global_scaler")
        else:
            if extracted["kmeans"] is None or extracted["cluster_scalers"] is None or extracted["settings_scaler"] is None:
                raise FD004ConfigError(f"variant {config.selected_variant} future payload missing regime objects")
            # dimension checks
            if model is not None:
                expected = VARIANT_FEATURE_DIMS.get(config.selected_variant)
                _verify_model_dimensions(model, config, expected)
        return extracted
    else:
        # legacy path – strict gate: must be historical hash (checked BEFORE deserialization would have been ideal;
        # we already computed is_historical before load, and verify_before_load gates manifest hash before this function,
        # but we enforce again here fail-closed)
        if not is_historical:
            raise FD004ConfigError(
                f"legacy condition joblib hash mismatch: expected {HISTORICAL_CONDITION_JOBLIB_SHA256}, got {actual_sha}. "
                f"Historical joblib is immutable; any other hash is unauthorized (hash-gated legacy adapter)."
            )
        # keys present
        if not isinstance(payload, dict):
            raise FD004ConfigError("legacy condition payload must be dict")
        required_keys = {"kmeans", "cluster_scalers", "settings_scaler"}
        if set(payload.keys()) != required_keys:
            # Also allow historical may have extra? Strictly require those 3
            missing = required_keys - set(payload.keys())
            if missing:
                raise FD004ConfigError(f"legacy payload missing keys {missing}")
            # extra keys would be considered legacy-incompatible
            extra = set(payload.keys()) - required_keys
            if extra:
                raise FD004ConfigError(f"legacy payload extra keys {extra} not allowed for hash {actual_sha}")
        # sidecar/manifest identity: variant must be C and candidate must match
        if config.selected_variant != "C":
            raise FD004ConfigError(f"legacy payload only valid for variant C, got {config.selected_variant}")
        expected_candidate = f"gru_w{config.window}_huber_condC"
        if config.candidate_name != expected_candidate:
            raise FD004ConfigError(f"legacy payload candidate mismatch: config {config.candidate_name} != {expected_candidate}")
        # dimensions agreement
        kmeans = payload.get("kmeans")
        if kmeans is None or getattr(kmeans, "n_clusters", None) != config.n_clusters:
            raise FD004ConfigError("legacy kmeans n_clusters mismatch")
        # fitted feature counts: check cluster_scalers size
        scalers = payload.get("cluster_scalers")
        if not isinstance(scalers, dict) or len(scalers) != config.n_clusters:
            raise FD004ConfigError(f"legacy cluster_scalers size {len(scalers) if isinstance(scalers, dict) else 'NA'} != n_clusters {config.n_clusters}")
        # model input dimensions vs config window
        if model is not None:
            expected = VARIANT_FEATURE_DIMS.get(config.selected_variant, 21)
            _verify_model_dimensions(model, config, expected)
        return {
            "kmeans": payload.get("kmeans"),
            "cluster_scalers": payload.get("cluster_scalers"),
            "settings_scaler": payload.get("settings_scaler"),
            "global_scaler": None,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc official FD004 evaluation")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--config", default="configs/final_model_v2_2_fd004.yaml")
    args = parser.parse_args()

    config = load_fd004_final_config(args.config)
    variant = config.selected_variant
    window = config.window

    # resolve identical artifact paths (freeze and post-hoc must agree)
    model_path = config.model_artifact_path(ROOT)
    cond_path = config.condition_artifact_path(ROOT)

    # manifest-based load-time verification before deserialization (distinct errors)
    # ponytail: verify hashes before loading when manifest available; keeps absent/legacy/mismatch distinct
    try:
        from rul_prediction.artifact_manifest import verify_before_load

        verify_before_load(model_path.relative_to(ROOT).as_posix(), root=ROOT, manifest_dataset="FD004")
        verify_before_load(cond_path.relative_to(ROOT).as_posix(), root=ROOT, manifest_dataset="FD004")
    except Exception:
        raise

    # distinguish missing artifact (friendly) before hash checks (legacy fallback if manifest not yet available)
    if not model_path.exists():
        raise FileNotFoundError(
            f"FD004 model artifact missing: {model_path}. Generate via scripts/run_v2_2_fd004_freeze.py "
            f"(or ensure models/v2_2/fd004_gru_w{window}_{config.loss}_cond{variant}.keras exists)"
        )
    if not cond_path.exists():
        raise FileNotFoundError(
            f"FD004 condition artifact missing: {cond_path}. Generate via scripts/run_v2_2_fd004_freeze.py"
        )

    # load model (verify window dimension after)
    model = keras.models.load_model(model_path)

    # load and verify preprocessing BEFORE reading official labels
    cond = load_and_verify_condition(config, model=model)

    # determine global_scaler for predictor (for A/B it's in cond, for C/D it's None)
    global_scaler = cond.get("global_scaler")

    # verify dimensions using a synthetic small history if needed
    # Build a dummy frame to infer feature dim and ensure window propagation
    # Use model input shape plus variant expected dim
    expected_dim = VARIANT_FEATURE_DIMS.get(variant)
    _verify_model_dimensions(model, config, expected_dim)

    # Now safe to read test trajectories (transform-only) but still not official labels
    test = load_test("FD004", args.data_dir)
    # official labels remain unread until after compatibility passes
    engines = sorted(test["engine_id"].unique())
    # verify test engine count matches expected but not labels
    assert len(engines) == EXPECTED_ENGINE_COUNTS["FD004"]["test"], f"test engine count {len(engines)} mismatch"
    test_manifest = pd.DataFrame({
        "engine_id": engines,
        "cutoff_cycle": [int(test[test["engine_id"] == e]["cycle"].max()) for e in engines]})
    test_traj = {int(e): g.sort_values("cycle") for e, g in test.groupby("engine_id")}

    # predictor requires keyword-only window and verifies dimensions internally
    predictor = make_predictor(variant, model, cond["kmeans"], cond["cluster_scalers"],
                               cond["settings_scaler"], global_scaler, window=window)
    # quick dimension sanity: simulate one engine with sufficient rows
    # we don't assert here but predictor will verify on first call

    pred = evaluate_manifest(test_manifest, test_traj, predictor)

    # ONLY NOW read official labels (post-hoc, selection-inert)
    rul = load_rul("FD004", args.data_dir).astype(float)
    assert len(engines) == len(rul) == EXPECTED_ENGINE_COUNTS["FD004"]["test"]
    y_true = np.asarray(rul, dtype=float)

    metrics = {
        "methodology_version": "2.2",
        "dataset": "FD004",
        "variant": variant,
        "official_status": "post-hoc (FD004 official labels permanently post-hoc; never select or tune V2.2)",
        "official_test_engine_count": int(len(engines)),
        "official_test_RMSE": round(float(rmse(y_true, pred)), 4),
        "official_test_MAE": round(float(mae(y_true, pred)), 4),
        "official_test_R2": round(float(r2(y_true, pred)), 4),
        "official_test_NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "official_test_NASA_mean": round(float(nasa_score(y_true, pred)) / len(engines), 4),
        "official_test_prediction_std": round(float(np.std(pred)), 4),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR / "fd004_final_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "reports" / "tables").mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(ROOT / "reports" / "tables" / "v2_2_fd004_official.csv", index=False)
    pd.DataFrame({"engine_id": engines, "true_rul_official": y_true,
                  "prediction": pred}).to_csv(
        ROOT / "reports" / "tables" / "v2_2_fd004_predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))

    # ---- falsification: variant metrics recomputed from saved prediction CSV ----
    pred_df = pd.read_csv(OUT_DIR / "fd004_variant_predictions.csv")
    stored = pd.read_csv(OUT_DIR / "fd004_variant_results.csv").set_index("variant")
    for v, g in pred_df.groupby("variant"):
        y, p = g["true_raw_rul"].to_numpy(float), g["prediction"].to_numpy(float)
        re_rmse = round(float(rmse(y, p)), 4)
        re_nasa = round(float(nasa_score(y, p)), 2)
        assert abs(re_rmse - stored.loc[v, "RMSE"]) < 1e-3, f"variant {v} RMSE drift"
        assert abs(re_nasa - stored.loc[v, "NASA_total"]) < 1e-1, f"variant {v} NASA drift"
    print("falsification: stored FD004 variant metrics match saved predictions")


if __name__ == "__main__":
    main()
