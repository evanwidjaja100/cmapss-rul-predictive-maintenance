"""Methodology V2.2: FD004 condition-aware variant comparison, training-control clean.

Repairs V2.2-8: the 37 outer validation engines must NEVER influence training
control. For every variant (A global / B global+settings / C regime scalers /
D C+settings+one-hot):

    Stage 1: fit ALL preprocessing (KMeans, cluster scalers, settings scaler,
             global scaler) on inner-fit (150 of the 175 training engines)
             ONLY; train GRU w45 huber with EarlyStopping monitored on
             inner-stop (25 engines) ONLY; record best_epoch.
    Stage 2: discard; refit preprocessing on ALL 175 training engines;
             retrain a FRESH model for exactly best_epoch with NO validation
             data; evaluate the 37 untouched validation engines on the fixed
             FD004 validation manifest.

Validation / official-test rows may only predict cluster + transform features,
never fit/refit/update. Variant selection: PRIMARY NASA per engine, SECONDARY
RMSE, then |signed bias| (pre-declared, V2_2_REPAIR_PLAN.md). Official FD004
labels remain POST-HOC forever and never select the variant.

Outputs:
    experiments/v2_2/fd004_variant_results.csv
    experiments/v2_2/fd004_variant_predictions.csv
    experiments/v2_2/fd004_best_epochs.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from rul_prediction.benchmark.fd004_config import (
    REPO_ROOT,
    compute_split_evidence_hashes,
)
from rul_prediction.data.condition import (
    SETTING_COLUMNS,
    condition_feature_matrix,
    fit_condition_models,
)
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.pseudo_test import load_manifest
from rul_prediction.data.v2_preprocessing import add_raw_rul, build_v2_train_sequences
from rul_prediction.data.windows import build_window, window_mask
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score
from rul_prediction.models.v2_models import v2_gru
from rul_prediction.reproducibility import (
    DirtyExecutionError,
    assert_reproducible_run_state,
)
from rul_prediction.training.trainer import set_seed

# ponytail: single explicit recipe drives training+config (no scattered literals)
FD004_RECIPE: dict = {
    "window": 45,
    "units": (128, 64),
    "dropout": 0.3,
    "loss": "huber",
    "optimizer": {"name": "adam", "clipnorm": 1.0},
    "learning_rate": 0.001,
    "batch_size": 256,
    "seed": 42,
    "n_clusters": 6,
    "clustering_seed": 42,
    "clustering_n_init": 10,
    "inner_seed": 4201,
    "inner_fit_size": 150,
    "inner_stop_size": 25,
    "epochs": 60,
    "patience": 8,
}
# legacy constants derived from recipe (keeps importers green)
WINDOW = FD004_RECIPE["window"]
OUT_DIR = REPO_ROOT / "experiments" / "v2_2"
VARIANTS = {"A", "B", "C", "D"}
INNER_SEED = FD004_RECIPE["inner_seed"]
INNER_FIT_SIZE = FD004_RECIPE["inner_fit_size"]
INNER_STOP_SIZE = FD004_RECIPE["inner_stop_size"]
EPOCHS = FD004_RECIPE["epochs"]
BATCH_SIZE = FD004_RECIPE["batch_size"]
PATIENCE = FD004_RECIPE["patience"]
SEED = FD004_RECIPE["seed"]


def inner_split(train_ids: set[int], n_fit: int = INNER_FIT_SIZE,
                n_stop: int = INNER_STOP_SIZE, seed: int = INNER_SEED,
                ) -> tuple[set[int], set[int]]:
    import random
    rng = random.Random(seed)
    ids = sorted(int(e) for e in train_ids)
    rng.shuffle(ids)
    inner_fit, inner_stop = set(ids[:n_fit]), set(ids[n_fit:n_fit + n_stop])
    if not inner_fit.isdisjoint(inner_stop) or inner_fit | inner_stop != set(ids):
        raise ValueError("inner split derivation invariant violated (overlap or incomplete partition)")
    return inner_fit, inner_stop


def build_matrix(variant: str, rows: pd.DataFrame, kmeans, cluster_scalers,
                 settings_scaler, global_scaler) -> np.ndarray:
    if variant in ("A", "B"):
        cols = SENSOR_COLUMNS + (SETTING_COLUMNS if variant == "B" else [])
        return global_scaler.transform(rows[cols].to_numpy(dtype=float)).astype(np.float32)
    matrix, _, _ = condition_feature_matrix(
        rows, kmeans, cluster_scalers, settings_scaler,
        with_settings=variant == "D", with_regime=variant == "D")
    return matrix


def make_predictor(variant: str, model, kmeans, cluster_scalers, settings_scaler,
                   global_scaler, *, window: int):
    """Require keyword-only window; verifies dimensions; uses for build_window/mask."""
    # ponytail: window is behavior-driving; caller must pass explicitly (no global fallback)
    if not isinstance(window, int) or window <= 0:
        raise ValueError(f"window must be positive int, got {window!r}")
    def predict_one(history, cutoff):
        rows = history.sort_values("cycle").reset_index(drop=True)
        # Fail fast when the model exposes no verifiable shape information;
        # dimension verification must never be silently bypassed.
        input_shape = getattr(model, "input_shape", None)
        shapes = [input_shape] if isinstance(input_shape, tuple) else (
            list(input_shape) if isinstance(input_shape, list) else None)
        if not shapes:
            raise ValueError(
                f"model exposes no usable input_shape {input_shape!r}; cannot verify window/feature dimensions"
            )
        features = build_matrix(variant, rows, kmeans, cluster_scalers, settings_scaler,
                                global_scaler)
        n_features = features.shape[1]
        # Verify model input shape vs config window/features.
        expected = shapes[0]
        if not (isinstance(expected, tuple) and len(expected) == 3):
            raise ValueError(
                f"model windowed-input shape {expected!r} is not (None, window, features)"
            )
        if expected[1] != window or expected[2] != n_features:
            raise ValueError(
                f"model input {expected} disagrees with window {window} features {n_features}"
            )
        win, n_obs, _ = build_window(features, len(rows), window)
        # window=47 produces (1,47,features) and (1,47) mask – verified by shape
        if win.shape != (window, n_features):
            raise ValueError(f"window tensor shape {win.shape} != ({window},{n_features})")
        mask = window_mask(n_obs, window)
        if mask.shape != (window,):
            raise ValueError(f"mask shape {mask.shape} != ({window},)")
        return float(model.predict([win[None], mask[None]],
                                   verbose=0)[0, 0])
    return predict_one


def fit_preprocessing(variant: str, frame: pd.DataFrame, allowed_ids: set[int],
                      k: int = 6, seed: int = 42, n_init: int = 10):
    """Fit preprocessing on `allowed_ids` rows only (asserted subset invariant).

    `k` / `seed` / `n_init` are the deployment-clustering hyperparameters; the
    final freeze path passes them explicitly from the YAML (never defaults).
    """
    if variant in ("A", "B"):
        cols = SENSOR_COLUMNS + (SETTING_COLUMNS if variant == "B" else [])
        global_scaler = StandardScaler().fit(
            frame[frame["engine_id"].isin(allowed_ids)][cols].to_numpy(dtype=float))
        return {"global_scaler": global_scaler, "kmeans": None,
                "cluster_scalers": None, "settings_scaler": None}
    kmeans, cluster_scalers, settings_scaler = fit_condition_models(
        frame, allowed_ids, k=k, seed=seed, n_init=n_init)
    return {"global_scaler": None, "kmeans": kmeans,
            "cluster_scalers": cluster_scalers, "settings_scaler": settings_scaler}


def train_epochs(variant: str, frame: pd.DataFrame, fit_ids: set[int],
                 pre: dict, epochs: int | None, validate: set[int] | None) -> tuple:
    """Train GRU on `fit_ids`; if `validate` is given, early-stop on it (stage 1)."""
    # ponytail: explicit recipe values; no hidden global fallback except via FD004_RECIPE
    window = FD004_RECIPE["window"]
    seed = FD004_RECIPE["seed"]
    loss = FD004_RECIPE["loss"]
    batch_size = FD004_RECIPE["batch_size"]
    units = FD004_RECIPE["units"]
    dropout = FD004_RECIPE["dropout"]
    lr = FD004_RECIPE["learning_rate"]
    # allow callers to override via pre? but use recipe for canonical
    train_rows = frame[frame["engine_id"].isin(fit_ids)].sort_values(["engine_id", "cycle"])
    X = build_matrix(variant, train_rows, pre["kmeans"], pre["cluster_scalers"],
                     pre["settings_scaler"], pre["global_scaler"])
    rul = add_raw_rul(train_rows)["rul"].to_numpy(dtype=np.float32)
    X_seq, y_seq, _, _, masks = build_v2_train_sequences(
        X, train_rows["engine_id"].to_numpy(), rul, window)
    set_seed(seed)
    model = v2_gru(window, X.shape[1], units=units, dropout=dropout, loss=loss, seed=seed,
                   learning_rate=lr, clipnorm=float(FD004_RECIPE["optimizer"]["clipnorm"]))
    callbacks = []
    if validate is not None:
        val_rows = frame[frame["engine_id"].isin(validate)].sort_values(["engine_id", "cycle"])
        X_val = build_matrix(variant, val_rows, pre["kmeans"], pre["cluster_scalers"],
                             pre["settings_scaler"], pre["global_scaler"])
        y_val = add_raw_rul(val_rows)["rul"].to_numpy(dtype=np.float32)
        X_val_seq, y_val_seq, _, _, m_val = build_v2_train_sequences(
            X_val, val_rows["engine_id"].to_numpy(), y_val, window)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        model.fit([X_seq, masks], y_seq, validation_data=([X_val_seq, m_val], y_val_seq),
                  batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=0)
        best_epoch = int(np.argmin(model.history.history["val_loss"]) + 1)
        return model, best_epoch, X.shape[1]
    model.fit([X_seq, masks], y_seq, batch_size=batch_size, epochs=epochs, verbose=0)
    return model, None, X.shape[1]


def run_variant(variant: str, frame: pd.DataFrame, train_ids, val_ids,
                splits_dir: str) -> tuple[dict, list[dict], dict]:
    print(f"== variant {variant} ==")
    inner_fit, inner_stop = inner_split(train_ids)
    if not (val_ids.isdisjoint(inner_fit) and val_ids.isdisjoint(inner_stop)):
        raise ValueError(f"variant {variant}: validation engines leak into inner split")
    if inner_fit | inner_stop != set(train_ids):
        raise ValueError(f"variant {variant}: inner split does not partition training engines")
    manifest = load_manifest(Path(splits_dir) / "fd004_v2_1_validation_cutoffs.csv")
    if len(manifest) != len(val_ids) * 5:
        raise ValueError(
            f"variant {variant}: validation manifest rows {len(manifest)} != expected {len(val_ids) * 5}"
        )
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(val_ids)].groupby("engine_id")
    }

    start = time.perf_counter()
    # Stage 1: preprocessing + early stopping on inner-fit / inner-stop only
    pre1 = fit_preprocessing(variant, frame, inner_fit, k=FD004_RECIPE["n_clusters"], seed=FD004_RECIPE["clustering_seed"], n_init=FD004_RECIPE["clustering_n_init"])
    _, best_epoch, n_inputs = train_epochs(variant, frame, inner_fit, pre1, EPOCHS, inner_stop)
    # Stage 2: preprocessing on ALL 175 training engines; fixed-duration retrain
    pre2 = fit_preprocessing(variant, frame, train_ids, k=FD004_RECIPE["n_clusters"], seed=FD004_RECIPE["clustering_seed"], n_init=FD004_RECIPE["clustering_n_init"])
    model, _, n_inputs2 = train_epochs(variant, frame, train_ids, pre2, best_epoch, None)
    if n_inputs != n_inputs2:
        raise ValueError(f"variant {variant}: feature dimension changed between stages ({n_inputs} vs {n_inputs2})")
    training_time = round(time.perf_counter() - start, 2)

    predictor = make_predictor(variant, model, pre2["kmeans"], pre2["cluster_scalers"],
                               pre2["settings_scaler"], pre2["global_scaler"], window=FD004_RECIPE["window"])
    pred = evaluate_manifest(manifest, trajectories, predictor)
    y_true = manifest["true_raw_rul"].to_numpy()
    per_engine = []
    for e, g in manifest.assign(prediction=pred).groupby("engine_id"):
        per_engine.append(nasa_score(g["true_raw_rul"].to_numpy(),
                                     g["prediction"].to_numpy()))
    row = {
        "variant": variant,
        "inputs": n_inputs2,
        "train_engine_count": len(train_ids),
        "validation_engine_count": len(val_ids),
        "inner_fit_engine_count": len(inner_fit),
        "inner_stop_engine_count": len(inner_stop),
        "best_epoch": best_epoch,
        "validation_sample_count": len(manifest),
        "RMSE": round(float(rmse(y_true, pred)), 4),
        "MAE": round(float(mae(y_true, pred)), 4),
        "R2": round(float(r2(y_true, pred)), 4),
        "NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "NASA_mean_per_engine": round(float(np.mean(per_engine)), 4),
        "signed_bias_mean": round(float(np.mean(pred - y_true)), 4),
        "prediction_std": round(float(np.std(pred)), 4),
        "training_time": training_time,
    }
    prediction_rows = [
        {"variant": variant, "engine_id": int(r.engine_id), "cutoff_cycle": int(r.cutoff_cycle),
         "true_raw_rul": float(r.true_raw_rul), "prediction": float(p)}
        for r, p in zip(manifest.itertuples(index=False), pred)
    ]
    print(f"variant {variant}: RMSE={row['RMSE']} R2={row['R2']} "
          f"NASA_per_engine={row['NASA_mean_per_engine']} "
          f"pred_std={row['prediction_std']} best_epoch={best_epoch}")
    return row, prediction_rows, {"variant": variant, "best_epoch": best_epoch,
                                   "inner_seed": INNER_SEED}


def main() -> None:
    parser = argparse.ArgumentParser(description="FD004 V2.2 variant comparison")
    parser.add_argument("--variants", nargs="*", default=["A", "B", "C", "D"])
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--splits-dir", default=str(REPO_ROOT / "experiments" / "splits"))
    parser.add_argument("--config-out", default=str(REPO_ROOT / "configs" / "final_model_v2_2_fd004.yaml"))
    parser.add_argument("--allow-dirty-reason", default=None,
                        help="nonempty reason permitting dirty execution inputs (requires --allow-dirty-snapshot-dir)")
    parser.add_argument("--allow-dirty-snapshot-dir", default=None,
                        help="durable destination for the dirty-source snapshot (required with --allow-dirty-reason)")
    args = parser.parse_args()

    # fail-closed reproducibility gate BEFORE any training/loading/output
    try:
        assert_reproducible_run_state(
            allow_dirty_execution=bool(args.allow_dirty_reason),
            dirty_reason=args.allow_dirty_reason,
            snapshot_dir=args.allow_dirty_snapshot_dir,
        )
    except DirtyExecutionError as e:
        sys.exit(f"refusing to run FD004 variant comparison: {e}")

    frame = load_train("FD004", args.data_dir)
    split = json.loads((Path(args.splits_dir) / "FD004_v2_seed42.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_engine_ids"])
    val_ids = set(split["validation_engine_ids"])
    cal_ids = set(split["calibration_engine_ids"])
    if not (len(train_ids) == 175 and len(val_ids) == 37 and len(cal_ids) == 37):
        raise ValueError(
            f"FD004 split counts dev/val/cal = {len(train_ids)}/{len(val_ids)}/{len(cal_ids)}; expected 175/37/37"
        )
    if not (val_ids.isdisjoint(train_ids) and cal_ids.isdisjoint(train_ids) and val_ids.isdisjoint(cal_ids)):
        raise ValueError("FD004 split engine-ID sets are not disjoint")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # check existing artifacts for resume/idempotency
    existing_results = pd.read_csv(OUT_DIR / "fd004_variant_results.csv") if (OUT_DIR / "fd004_variant_results.csv").exists() else pd.DataFrame()
    if not existing_results.empty:
        if existing_results["variant"].duplicated().any():
            raise ValueError("fd004_variant_results.csv contains duplicate variant rows – refusing to continue")
    existing_preds = pd.read_csv(OUT_DIR / "fd004_variant_predictions.csv") if (OUT_DIR / "fd004_variant_predictions.csv").exists() else pd.DataFrame()
    existing_best = pd.read_csv(OUT_DIR / "fd004_best_epochs.csv") if (OUT_DIR / "fd004_best_epochs.csv").exists() else pd.DataFrame()
    if not existing_best.empty and existing_best["variant"].duplicated().any():
        raise ValueError("fd004_best_epochs.csv duplicate variants")

    best_rows: list[dict] = []
    for variant in args.variants:
        if variant not in VARIANTS:
            sys.exit(f"unknown variant {variant}")
        done = set(existing_results["variant"]) if not existing_results.empty else set()
        if variant in done:
            print(f"skip variant {variant} (already done)")
            continue
        row, pred_rows, control = run_variant(variant, frame, train_ids, val_ids,
                                              args.splits_dir)
        # validate 37*5 predictions before writing
        if len(pred_rows) != 37 * 5:
            raise ValueError(f"variant {variant} predictions {len(pred_rows)} != 185")
        # atomic append for results
        path = OUT_DIR / "fd004_variant_results.csv"
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        # append predictions – reject duplicates across file
        pred_path = OUT_DIR / "fd004_variant_predictions.csv"
        if pred_path.exists():
            already = set(zip(pd.read_csv(pred_path)["variant"], pd.read_csv(pred_path)["engine_id"], pd.read_csv(pred_path)["cutoff_cycle"]))
            new_keys = [(r["variant"], r["engine_id"], r["cutoff_cycle"]) for r in pred_rows]
            dupes = [k for k in new_keys if k in already]
            if dupes:
                raise ValueError(f"duplicate prediction rows for {variant}: {dupes[:3]}")
        pd.DataFrame(pred_rows).to_csv(pred_path, mode="a",
                                       header=not pred_path.exists(), index=False)
        best_rows.append(control)
        # update in-memory tracking for subsequent loop iterations
        existing_results = pd.concat([existing_results, pd.DataFrame([row])], ignore_index=True)
        existing_preds = pd.concat([existing_preds, pd.DataFrame(pred_rows)], ignore_index=True) if not existing_preds.empty else pd.DataFrame(pred_rows)

    # merge resumed best-epoch rows by variant instead of overwrite
    if best_rows:
        merged: dict[str, dict] = {}
        if not existing_best.empty:
            for _, r in existing_best.iterrows():
                merged[str(r["variant"])] = {"variant": str(r["variant"]), "best_epoch": int(r["best_epoch"]), "inner_seed": int(r["inner_seed"])}
        for c in best_rows:
            if c["variant"] in merged:
                # prior file already had this variant and it was not retrained this run;
                # overwriting would corrupt the historical control row
                raise ValueError(f"best_epochs merge collision for variant {c['variant']}")
            merged[c["variant"]] = c
        out_df = pd.DataFrame(sorted(merged.values(), key=lambda x: x["variant"]))
        # atomic write
        tmp = OUT_DIR / "fd004_best_epochs.csv.tmp"
        out_df.to_csv(tmp, index=False)
        tmp.replace(OUT_DIR / "fd004_best_epochs.csv")

    # final validation: predictions per variant exactly 185, ids disjoint etc.
    if (OUT_DIR / "fd004_variant_predictions.csv").exists():
        dfp = pd.read_csv(OUT_DIR / "fd004_variant_predictions.csv")
        for v, g in dfp.groupby("variant"):
            if len(g) != 37 * 5:
                raise ValueError(f"variant {v} prediction count {len(g)} != 185")
            if g.duplicated(subset=["variant", "engine_id", "cutoff_cycle"]).any():
                raise ValueError(f"duplicate rows for variant {v}")

    results = pd.read_csv(OUT_DIR / "fd004_variant_results.csv") if (OUT_DIR / "fd004_variant_results.csv").exists() else pd.DataFrame()
    # require at least the requested variants present
    if not results.empty:
        req = set(args.variants)
        have = set(results["variant"])
        if not req.issubset(have):
            # partial – do not overwrite canonical config
            print(f"partial results {have} missing {req - have}; deferring config write")

    if not results.empty:
        # decide on config writing: refuse canonical path unless all 4 variants present.
        # Compare RESOLVED absolute paths so absolute/backward-slashed invocations
        # cannot bypass the canonical guard (Section 9.6).
        canonical_path = (REPO_ROOT / "configs" / "final_model_v2_2_fd004.yaml").resolve()
        is_canonical = Path(args.config_out).resolve() == canonical_path
        if is_canonical and set(results["variant"]) != VARIANTS:
            print(f"refusing to overwrite canonical config {args.config_out}: need {VARIANTS}, have {set(results['variant'])}", file=sys.stderr)
            # do not write; exit gracefully but inform
            print("\nFD004 variant results (37 held-out validation engines):")
            print(results.sort_values("NASA_mean_per_engine").to_string(index=False))
            return
        print("\nFD004 variant results (37 held-out validation engines):")
        print(results.sort_values("NASA_mean_per_engine").to_string(index=False))
        ranked = results.copy()
        ranked["abs_signed_bias_mean"] = ranked["signed_bias_mean"].abs()
        winner = ranked.sort_values(["NASA_mean_per_engine", "RMSE",
                                     "abs_signed_bias_mean"]).iloc[0]
        print(f"\npre-declared rule (NASA per engine, then RMSE, then |bias|) selects: {winner['variant']}")
        best_rows_df = pd.read_csv(OUT_DIR / "fd004_best_epochs.csv") \
            if (OUT_DIR / "fd004_best_epochs.csv").exists() else pd.DataFrame()
        write_fd004_config(winner, results, best_rows_df, args.config_out)
    else:
        print("no variant results yet; config not written")


def write_fd004_config(winner: pd.Series, results: pd.DataFrame,
                       best_rows: pd.DataFrame, config_out: str) -> None:
    """Generate configs/final_model_v2_2_fd004.yaml from the variant results.

    Uses the single authoritative recipe FD004_RECIPE; no scattered literals.
    """
    import importlib.metadata as md
    import yaml
    from rul_prediction.data.canonical_hash import canonical_sha256_json

    # validation: refuse unless all 4 rows exist for canonical path.
    # Resolved-path comparison so absolute/alternative spellings cannot bypass.
    canonical_path = (REPO_ROOT / "configs" / "final_model_v2_2_fd004.yaml").resolve()
    if Path(config_out).resolve() == canonical_path:
        if set(results["variant"]) != VARIANTS:
            raise ValueError(f"canonical config requires {VARIANTS}, got {set(results['variant'])}")

    winner_row = results[results.variant == winner["variant"]].iloc[0]
    best_epoch = int(winner_row["best_epoch"])
    # cross-check best_epoch vs best_rows if available
    if not best_rows.empty and winner["variant"] in set(best_rows["variant"]):
        br = best_rows[best_rows.variant == winner["variant"]].iloc[0]
        if int(br["best_epoch"]) != best_epoch:
            raise ValueError(f"best_epoch mismatch winner {best_epoch} vs best_rows {br['best_epoch']}")

    versions = {p: md.version(p) for p in ("tensorflow", "numpy", "pandas",
                                           "scikit-learn", "xgboost", "joblib")}
    versions["python"] = sys.version.split()[0]
    split = json.loads((REPO_ROOT / "experiments" / "splits" / "FD004_v2_seed42.json").read_text(encoding="utf-8"))
    # raw exact-file evidence hashes for the split JSON and validation cutoff CSV
    raw_hashes = compute_split_evidence_hashes(
        "experiments/splits/FD004_v2_seed42.json",
        "experiments/splits/fd004_v2_1_validation_cutoffs.csv",
        root=REPO_ROOT,
    )
    # recipe-driven values
    cfg = {
        "methodology_version": "2.2",
        "dataset": "FD004",
        "target": "raw RUL regression under multiple operating regimes",
        "model": {
            "candidate_name": f"gru_w{FD004_RECIPE['window']}_huber_cond{winner['variant']}",
            "architecture": "gru",
            "window": int(FD004_RECIPE["window"]),
            "units": list(FD004_RECIPE["units"]),
            "dropout": float(FD004_RECIPE["dropout"]),
            "loss": str(FD004_RECIPE["loss"]),
            "optimizer": {"name": str(FD004_RECIPE["optimizer"]["name"]), "clipnorm": float(FD004_RECIPE["optimizer"]["clipnorm"])},
            "learning_rate": float(FD004_RECIPE["learning_rate"]),
            "batch_size": int(FD004_RECIPE["batch_size"]),
            "seed": int(FD004_RECIPE["seed"]),
            "fixed_epochs": int(best_epoch),
        },
        "condition_preprocessing": {
            "variant": str(winner["variant"]),
            "clustering": {
                "method": "kmeans",
                "n_clusters": int(FD004_RECIPE["n_clusters"]),
                "random_state": int(FD004_RECIPE["clustering_seed"]),
                "n_init": int(FD004_RECIPE["clustering_n_init"]),
            },
            "operating_setting_columns": list(SETTING_COLUMNS),
            "sensor_scaling": {
                "mode": "per_regime_standard_scaler",
                "fit_scope": "training_rows_only",
            },
            "scopes": {
                "stage1_inner": "150 inner-fit / 25 inner-stop, seed 4201 (development engines only)",
                "stage2_refit": "175 development/training engines (variant comparison; validation transform-only)",
                "final_freeze": "212 engines (175 training + 37 validation; 37 calibration reserved)",
                "reserved_calibration": "37 engines (held out from all fitting; transform-only)",
                "official_test": "transform-only (never fit/refit/update; labels post-hoc)",
            },
            "fit_ids_stage2": "175 development/training engines only; validation and calibration rows transform-only",
            "official_test_rows": "transform-only (never fit/refit/update)",
        },
        "splits": {
            "provenance": "experiments/splits/FD004_v2_seed42.json",
            "development_engine_count": 175,
            "validation_engine_count": 37,
            "calibration_engine_count": 37,
            "inner_split": "150 inner-fit / 25 inner-stop, seed 4201",
            "development_engine_ids_sha256": canonical_sha256_json(sorted(split["train_engine_ids"])),
            "validation_engine_ids_sha256": canonical_sha256_json(sorted(split["validation_engine_ids"])),
            "calibration_engine_ids_sha256": canonical_sha256_json(sorted(split["calibration_engine_ids"])),
            # raw exact-file digests (CRLF->LF normalized; never compared to canonical digests)
            "split_provenance_file_sha256": raw_hashes["split_provenance_file_sha256"],
            "validation_cutoff_manifest_file_sha256": raw_hashes["validation_cutoff_manifest_file_sha256"],
            "validation_manifest": "experiments/splits/fd004_v2_1_validation_cutoffs.csv",
        },
        "training": {
            "final_engine_count": 212,
            "reserved_engine_count": 37,
            "epoch_selection_rule": "final_epoch_count = best epoch from inner-fit/inner-stop (development engines only)",
            "validation_data_in_final_fit": False,
        },
        "selection_policy": "PRIMARY lowest NASA per engine; SECONDARY lower RMSE; then smaller |signed bias| (validation engines only; official FD004 labels are POST-HOC and never select)",
        "variant_results": {
            str(r["variant"]): {"RMSE": float(r["RMSE"]), "R2": float(r["R2"]),
                                "NASA_mean_per_engine": float(r["NASA_mean_per_engine"]),
                                "signed_bias_mean": float(r["signed_bias_mean"]),
                                "best_epoch": int(r["best_epoch"])}
            for _, r in results.iterrows()
        },
        "software_versions": versions,
    }
    out = Path(config_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # atomic write with a unique temp name in the target directory
    import tempfile, os
    fd, tmp_name = tempfile.mkstemp(dir=str(out.parent), suffix=".yaml.tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False))
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(f"wrote {out} (variant {winner['variant']}, best_epoch {best_epoch})")


if __name__ == "__main__":
    main()
