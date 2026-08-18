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
from rul_prediction.training.trainer import set_seed

WINDOW = 45
OUT_DIR = Path("experiments/v2_2")
VARIANTS = {"A", "B", "C", "D"}
INNER_SEED = 4201          # same documented scheme: random.Random(4200 + 1)
INNER_FIT_SIZE = 150       # of 175 training engines
INNER_STOP_SIZE = 25
EPOCHS = 60
BATCH_SIZE = 256
PATIENCE = 8
SEED = 42


def inner_split(train_ids: set[int], n_fit: int = INNER_FIT_SIZE,
                n_stop: int = INNER_STOP_SIZE, seed: int = INNER_SEED,
                ) -> tuple[set[int], set[int]]:
    import random
    rng = random.Random(seed)
    ids = sorted(int(e) for e in train_ids)
    rng.shuffle(ids)
    inner_fit, inner_stop = set(ids[:n_fit]), set(ids[n_fit:n_fit + n_stop])
    assert inner_fit.isdisjoint(inner_stop)
    assert inner_fit | inner_stop == set(ids)
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
                   global_scaler):
    def predict_one(history, cutoff):
        rows = history.sort_values("cycle").reset_index(drop=True)
        features = build_matrix(variant, rows, kmeans, cluster_scalers, settings_scaler,
                                global_scaler)
        win, n_obs, _ = build_window(features, len(rows), WINDOW)
        return float(model.predict([win[None], window_mask(n_obs, WINDOW)[None]],
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
    train_rows = frame[frame["engine_id"].isin(fit_ids)].sort_values(["engine_id", "cycle"])
    X = build_matrix(variant, train_rows, pre["kmeans"], pre["cluster_scalers"],
                     pre["settings_scaler"], pre["global_scaler"])
    rul = add_raw_rul(train_rows)["rul"].to_numpy(dtype=np.float32)
    X_seq, y_seq, _, _, masks = build_v2_train_sequences(
        X, train_rows["engine_id"].to_numpy(), rul, WINDOW)
    set_seed(SEED)
    model = v2_gru(WINDOW, X.shape[1], loss="huber", seed=SEED)
    callbacks = []
    if validate is not None:
        val_rows = frame[frame["engine_id"].isin(validate)].sort_values(["engine_id", "cycle"])
        X_val = build_matrix(variant, val_rows, pre["kmeans"], pre["cluster_scalers"],
                             pre["settings_scaler"], pre["global_scaler"])
        y_val = add_raw_rul(val_rows)["rul"].to_numpy(dtype=np.float32)
        X_val_seq, y_val_seq, _, _, m_val = build_v2_train_sequences(
            X_val, val_rows["engine_id"].to_numpy(), y_val, WINDOW)
        callbacks = [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True)]
        model.fit([X_seq, masks], y_seq, validation_data=([X_val_seq, m_val], y_val_seq),
                  batch_size=BATCH_SIZE, epochs=epochs, callbacks=callbacks, verbose=0)
        best_epoch = int(np.argmin(model.history.history["val_loss"]) + 1)
        return model, best_epoch, X.shape[1]
    model.fit([X_seq, masks], y_seq, batch_size=BATCH_SIZE, epochs=epochs, verbose=0)
    return model, None, X.shape[1]


def run_variant(variant: str, frame: pd.DataFrame, train_ids, val_ids,
                splits_dir: str) -> tuple[dict, list[dict], dict]:
    print(f"== variant {variant} ==")
    inner_fit, inner_stop = inner_split(train_ids)
    assert val_ids.isdisjoint(inner_fit) and val_ids.isdisjoint(inner_stop)
    assert inner_fit | inner_stop == train_ids
    manifest = load_manifest(Path(splits_dir) / "fd004_v2_1_validation_cutoffs.csv")
    assert len(manifest) == len(val_ids) * 5
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(val_ids)].groupby("engine_id")
    }

    start = time.perf_counter()
    # Stage 1: preprocessing + early stopping on inner-fit / inner-stop only
    pre1 = fit_preprocessing(variant, frame, inner_fit)
    _, best_epoch, n_inputs = train_epochs(variant, frame, inner_fit, pre1, EPOCHS, inner_stop)
    # Stage 2: preprocessing on ALL 175 training engines; fixed-duration retrain
    pre2 = fit_preprocessing(variant, frame, train_ids)
    model, _, n_inputs2 = train_epochs(variant, frame, train_ids, pre2, best_epoch, None)
    assert n_inputs == n_inputs2
    training_time = round(time.perf_counter() - start, 2)

    predictor = make_predictor(variant, model, pre2["kmeans"], pre2["cluster_scalers"],
                               pre2["settings_scaler"], pre2["global_scaler"])
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
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--splits-dir", default="experiments/splits")
    parser.add_argument("--config-out", default="configs/final_model_v2_2_fd004.yaml")
    args = parser.parse_args()

    frame = load_train("FD004", args.data_dir)
    split = json.loads((Path(args.splits_dir) / "fd004_v2_seed42.json").read_text(encoding="utf-8"))
    train_ids = set(split["train_engine_ids"])
    val_ids = set(split["validation_engine_ids"])
    cal_ids = set(split["calibration_engine_ids"])
    assert len(train_ids) == 175 and len(val_ids) == 37 and len(cal_ids) == 37
    assert val_ids.isdisjoint(train_ids) and cal_ids.isdisjoint(train_ids)
    assert val_ids.isdisjoint(cal_ids)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_rows = []
    for variant in args.variants:
        if variant not in VARIANTS:
            sys.exit(f"unknown variant {variant}")
        done = set()
        path = OUT_DIR / "fd004_variant_results.csv"
        if path.exists():
            done = set(pd.read_csv(path)["variant"])
        if variant in done:
            print(f"skip variant {variant} (already done)")
            continue
        row, pred_rows, control = run_variant(variant, frame, train_ids, val_ids,
                                              args.splits_dir)
        pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
        pred_path = OUT_DIR / "fd004_variant_predictions.csv"
        pd.DataFrame(pred_rows).to_csv(pred_path, mode="a",
                                       header=not pred_path.exists(), index=False)
        best_rows.append(control)
    if best_rows:
        pd.DataFrame(best_rows).to_csv(OUT_DIR / "fd004_best_epochs.csv", index=False)

    results = pd.read_csv(OUT_DIR / "fd004_variant_results.csv")
    assert set(results["variant"]) == VARIANTS or set(results["variant"]) == set(args.variants)
    print("\nFD004 variant results (37 held-out validation engines):")
    print(results.sort_values("NASA_mean_per_engine").to_string(index=False))
    ranked = results.copy()
    ranked["abs_signed_bias_mean"] = ranked["signed_bias_mean"].abs()
    winner = ranked.sort_values(["NASA_mean_per_engine", "RMSE",
                                 "abs_signed_bias_mean"]).iloc[0]
    print(f"\npre-declared rule (NASA per engine, then RMSE, then |bias|) selects: {winner['variant']}")
    best_rows = pd.read_csv(OUT_DIR / "fd004_best_epochs.csv") \
        if (OUT_DIR / "fd004_best_epochs.csv").exists() else pd.DataFrame()
    write_fd004_config(winner, results, best_rows, args.config_out)


def write_fd004_config(winner: pd.Series, results: pd.DataFrame,
                       best_rows: pd.DataFrame, config_out: str) -> None:
    """Generate configs/final_model_v2_2_fd004.yaml from the variant results."""
    import importlib.metadata as md
    import yaml
    from rul_prediction.data.canonical_hash import canonical_sha256_json

    winner_row = results[results.variant == winner["variant"]].iloc[0]
    best_epoch = int(winner_row["best_epoch"])
    versions = {p: md.version(p) for p in ("tensorflow", "numpy", "pandas",
                                           "scikit-learn", "xgboost", "joblib")}
    import sys
    versions["python"] = sys.version.split()[0]
    split = json.loads((Path("experiments/splits") / "fd004_v2_seed42.json").read_text(encoding="utf-8"))
    cfg = {
        "methodology_version": "2.2",
        "dataset": "FD004",
        "target": "raw RUL regression under multiple operating regimes",
        "model": {
            "candidate_name": f"gru_w{WINDOW}_huber_cond{winner['variant']}",
            "architecture": "gru",
            "window": WINDOW,
            "units": [128, 64],
            "dropout": 0.3,
            "loss": "huber",
            "optimizer": "Adam(clipnorm=1.0)",
            "learning_rate": 0.001,
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "fixed_epochs": best_epoch,
        },
        "condition_preprocessing": {
            "variant": winner["variant"],
            "clustering": {
                "method": "kmeans",
                "n_clusters": 6,
                "random_state": 42,
                "n_init": 10,
            },
            "operating_setting_columns": list(SETTING_COLUMNS),
            "sensor_scaling": {
                "mode": "per_regime_standard_scaler",
                "fit_scope": "training_rows_only",
            },
            "fit_ids_stage2": "175 development/training engines only; validation and calibration rows transform-only",
            "official_test_rows": "transform-only (never fit/refit/update)",
        },
        "splits": {
            "provenance": "experiments/splits/fd004_v2_seed42.json",
            "development_engine_count": 175,
            "validation_engine_count": 37,
            "calibration_engine_count": 37,
            "inner_split": "150 inner-fit / 25 inner-stop, seed 4201",
            "development_engine_ids_sha256": canonical_sha256_json(sorted(split["train_engine_ids"])),
            "validation_engine_ids_sha256": canonical_sha256_json(sorted(split["validation_engine_ids"])),
            "calibration_engine_ids_sha256": canonical_sha256_json(sorted(split["calibration_engine_ids"])),
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
    out.write_text(yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    print(f"wrote {out} (variant {winner['variant']}, best_epoch {best_epoch})")


if __name__ == "__main__":
    main()