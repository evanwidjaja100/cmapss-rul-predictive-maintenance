"""Methodology M3 nested engine-group CV (FD001) — training-control clean.

Repairs M3-1/M3-2/M3-3 (see M3_REPAIR_PLAN.md):

- 85 development engines; 15 calibration engines NEVER enter any training
  control path (weights, epochs, iteration count, preprocessing fitting).
- 5 outer engine folds (seed 42, from the M2 manifest): per fold
  68 outer-training / 17 untouched outer-evaluation engines. The 17 outer-
  evaluation engines never influence fitting, early stopping, checkpoints or
  preprocessing.
- Inner early-stop split INSIDE the 68 outer-training engines:
  `random.Random(4200 + fold)` -> first 58 = inner-fit, last 10 = inner-stop
  (seeds: fold1=4201 ... fold5=4205; 58/10 ~ 85/15). Documented in
  M3_REPAIR_PLAN.md before any M3 result was inspected.
- Stage 1 (control): fit preprocessing on inner-fit ONLY, train with early
  stopping monitored on inner-stop ONLY; record best_epoch (deep) or
  best_iteration (XGBoost).
- Stage 2 (evaluation): discard stage-1 model; refit preprocessing on ALL 68
  outer-training engines; retrain a FRESH model for exactly best_epoch /
  best_iteration+1 with NO validation data; evaluate the 17 untouched
  outer-evaluation engines on the fixed outer pseudo-test manifest.
- Random forest has no early stopping: one stage, fit on all 68, evaluate 17.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rul_prediction.benchmark.m1 import (
    ROOT,
    classical_features,
    make_predictor,
    partition_sequences,
)
from rul_prediction.data.loader import SENSOR_COLUMNS, load_train
from rul_prediction.data.preprocessing import transform
from rul_prediction.data.pseudo_test import M2_LIFECYCLE_FRACTIONS, load_manifest
from rul_prediction.data.m2_splits import read_m2_cv_manifest
from rul_prediction.data.m1_preprocessing import add_raw_rul, build_m1_train_sequences
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

CV_CANDIDATES = [
    {"id": "gru_w45_huber", "model": "gru", "window": 45, "overrides": {"loss": "huber"}},
    {"id": "gru_w60_huber", "model": "gru", "window": 60, "overrides": {"loss": "huber"}},
    {"id": "lstm_w45_huber", "model": "lstm", "window": 45, "overrides": {"loss": "huber"}},
    {"id": "lstm_w60_huber", "model": "lstm", "window": 60, "overrides": {"loss": "huber"}},
    {"id": "rf_w60", "model": "rf", "window": 60, "overrides": None},
    {"id": "rf_w90", "model": "rf", "window": 90, "overrides": None},
    {"id": "xgb_w60_d6", "model": "xgboost", "window": 60, "overrides": {"max_depth": 6}},
    {"id": "xgb_w90_d6", "model": "xgboost", "window": 90, "overrides": {"max_depth": 6}},
]

INNER_FIT_SIZE = 58  # of 68 outer-training engines
INNER_STOP_SIZE = 10
INNER_SEED_BASE = 4200  # inner seed = INNER_SEED_BASE + fold (fold1=4201 ... fold5=4205)
SEED = 42


def inner_early_stop_split(outer_train_ids: set[int], fold: int,
                           n_fit: int = INNER_FIT_SIZE,
                           n_stop: int = INNER_STOP_SIZE) -> tuple[set[int], set[int]]:
    """Deterministic inner split of the outer-training engines (seeds 4201..4205).

    Documented scheme (M3_REPAIR_PLAN.md): random.Random(4200+fold) shuffles
    the sorted outer-training IDs; first n_fit = inner-fit, last n_stop = inner-stop.
    """
    rng = random.Random(INNER_SEED_BASE + fold)
    ids = sorted(int(e) for e in outer_train_ids)
    rng.shuffle(ids)
    inner_fit = set(ids[:n_fit])
    inner_stop = set(ids[n_fit:n_fit + n_stop])
    assert len(inner_fit) == n_fit and len(inner_stop) == n_stop
    assert inner_fit.isdisjoint(inner_stop)
    assert inner_fit | inner_stop == set(ids)
    return inner_fit, inner_stop


def load_m3_cv_artifacts(dataset: str = "FD001", data_dir: str | Path = "data/raw",
                           splits_dir: str | Path = "experiments/splits") -> dict:
    """Frame + M2 CV manifest payload (dev/cal split + 5 outer folds)."""
    frame = load_train(dataset, data_dir)
    payload = read_m2_cv_manifest(
        Path(splits_dir) / f"{dataset.lower()}_m2_group_cv_seed{SEED}.json")
    dev_ids = set(payload["development_engine_ids"])
    cal_ids = set(payload["calibration_engine_ids"])
    folds = [
        {"fold": f["fold"], "outer_train": set(f["training_engine_ids"]),
         "outer_eval": set(f["validation_engine_ids"])}
        for f in payload["folds"]
    ]
    assert len(folds) == 5
    assert dev_ids.isdisjoint(cal_ids)
    assert set().union(*(f["outer_train"] | f["outer_eval"] for f in folds)) == dev_ids
    return {"frame": frame, "folds": folds, "dev_ids": dev_ids, "cal_ids": cal_ids,
            "payload": payload}


def fold_scaler(frame, train_ids, fold: int, seed: int = SEED) -> StandardScaler:
    """Scaler fit on the fold's permitted training rows only."""
    rows = frame[frame["engine_id"].isin(train_ids)]
    return StandardScaler().fit(rows[SENSOR_COLUMNS].to_numpy(dtype=float))


def _training_data(frame, train_ids, scaler, window):
    rows = frame[frame["engine_id"].isin(train_ids)]
    X, y, ids, n_observed, masks = build_m1_train_sequences(
        transform(rows, SENSOR_COLUMNS, scaler),
        rows["engine_id"].to_numpy(), add_raw_rul(rows)["rul"].to_numpy(dtype=np.float32),
        window)
    F_train, _ = classical_features(X, ids, n_observed, window)
    return X, y, ids, n_observed, masks, F_train


def _stage1_deep(candidate, frame, inner_fit, inner_stop, fold, *,
                 epochs: int = 60, batch_size: int = 256, patience: int = 8,
                 seed: int = SEED) -> tuple[int, dict]:
    """Stage 1: scaler+model on inner-fit, early stopping on inner-stop -> best_epoch."""
    from tensorflow import keras

    from rul_prediction.models.m1_models import m1_gru, m1_lstm
    from rul_prediction.training.trainer import set_seed

    window = candidate["window"]
    scaler = fold_scaler(frame, inner_fit, fold, seed)
    X, y, _, _, masks, _ = _training_data(frame, inner_fit, scaler, window)
    X_stop, y_stop, _, _, m_stop = partition_sequences(frame, inner_stop, scaler, window)
    loss = candidate["overrides"].get("loss", "mse")
    builder = m1_gru if candidate["model"] == "gru" else m1_lstm
    set_seed(seed)
    model = builder(window, X.shape[2], loss=loss, seed=seed)
    history = model.fit(
        [X, masks], y, validation_data=([X_stop, m_stop], y_stop),
        batch_size=batch_size, epochs=epochs,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True)],
        verbose=0)
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    meta = {"best_epoch": best_epoch,
            "inner_fit_count": len(inner_fit), "inner_stop_count": len(inner_stop),
            "inner_seed": INNER_SEED_BASE + fold}
    return best_epoch, meta


def _stage1_xgboost(candidate, frame, inner_fit, inner_stop, fold, *,
                    seed: int = SEED) -> tuple[int, dict]:
    """Stage 1: features on inner-fit, best_iteration from eval on inner-stop."""
    from rul_prediction.models.xgboost_model import xgboost_regressor

    window = candidate["window"]
    scaler = fold_scaler(frame, inner_fit, fold, seed)
    X, y, ids, n_observed, masks, F_fit = _training_data(frame, inner_fit, scaler, window)
    X_stop, y_stop, ids_stop, n_stop_obs, m_stop = partition_sequences(
        frame, inner_stop, scaler, window)
    F_stop, _ = classical_features(X_stop, ids_stop, n_stop_obs, window)
    model = xgboost_regressor(seed)
    model.set_params(max_depth=candidate["overrides"].get("max_depth", 6))
    model.fit(F_fit, y, eval_set=[(F_stop, y_stop)], verbose=False)
    best_iteration = int(model.best_iteration)
    meta = {"best_iteration": best_iteration,
            "inner_fit_count": len(inner_fit), "inner_stop_count": len(inner_stop),
            "inner_seed": INNER_SEED_BASE + fold}
    return best_iteration, meta


def _stage2_fit(candidate, frame, outer_train, control: dict, fold: int, *,
                batch_size: int = 256, seed: int = SEED):
    """Stage 2: preprocessing on ALL outer-training engines; fixed-duration fit, NO validation."""
    from rul_prediction.models.xgboost_model import xgboost_regressor
    from rul_prediction.models.m1_models import m1_gru, m1_lstm
    from rul_prediction.training.trainer import set_seed

    window = candidate["window"]
    scaler = fold_scaler(frame, outer_train, fold, seed)
    X, y, ids, n_observed, masks, F_train = _training_data(frame, outer_train, scaler, window)
    if candidate["model"] == "rf":
        from rul_prediction.models.baseline import random_forest
        model = random_forest(seed).fit(F_train, y)
        return model, scaler, "n_estimators=300", X.shape[2]
    if candidate["model"] == "xgboost":
        n_est = control["best_iteration"] + 1
        model = xgboost_regressor(seed)
        model.set_params(max_depth=candidate["overrides"].get("max_depth", 6),
                         n_estimators=n_est, early_stopping_rounds=None)
        model.fit(F_train, y, verbose=False)
        return model, scaler, f"n_estimators={n_est},max_depth={candidate['overrides'].get('max_depth', 6)}", X.shape[2]
    loss = candidate["overrides"].get("loss", "mse")
    builder = m1_gru if candidate["model"] == "gru" else m1_lstm
    set_seed(seed)
    model = builder(window, X.shape[2], loss=loss, seed=seed)
    model.fit([X, masks], y, batch_size=batch_size, epochs=control["best_epoch"], verbose=0)
    return model, scaler, f"{builder.__name__};loss={loss};bs={batch_size};fixed_epochs={control['best_epoch']}", X.shape[2]


def git_provenance(root: str | Path | None = None) -> dict:
    """Git provenance at run start (delegates to rul_prediction.reproducibility).

    Reports Git HEAD, whole-repo and execution-scope dirty flags, status/diff
    hashes, deterministic source_tree_hash via git ls-files, and UTC timestamp.
    Uses NUL-delimited status and binary diffs; tracked content-hash uses
    domain-separated length-delimited encoding cmapss-tracked-source-v1.
    """
    from rul_prediction.reproducibility import collect_git_provenance

    return collect_git_provenance(root=root)


def source_tree_hash(root: str | Path | None = None) -> str:
    """Deterministic Git-tracked execution-input hash (reproducibility).

    Enumerates tracked execution inputs via git ls-files -z (src/**, scripts/**,
    configs/**, app_m1.py, .github/workflows/**, pyproject.toml, requirements*),
    sorts POSIX paths, and hashes with domain-separated length-delimited format
    cmapss-tracked-source-v1 (path length, path bytes, content length, content).
    Ignored/cached/generated files do not affect the hash. Fails closed: errors
    propagate from the canonical implementation instead of degrading to None.
    """
    from rul_prediction.reproducibility import tracked_source_tree_details

    return tracked_source_tree_details(root)["source_tree_hash"]


def run_metadata(dataset: str, candidate: str, fold: int, control: dict,
                 dev_ids, cal_ids, window: int, extra: dict | None = None) -> dict:
    """Reproducibility metadata for one candidate-fold run (M3-14)."""
    import importlib.metadata as md

    versions = {}
    for pkg in ("tensorflow", "numpy", "pandas", "scikit-learn", "xgboost", "joblib"):
        try:
            versions[pkg] = md.version(pkg)
        except Exception:
            versions[pkg] = "unknown"
    from rul_prediction.data.canonical_hash import canonical_sha256_json
    meta = {
        **git_provenance(),
        "source_tree_hash": source_tree_hash(),
        "methodology": "m3",
        "dataset": dataset,
        "candidate": candidate,
        "window": window,
        "outer_fold": fold,
        "inner_seed": control.get("inner_seed"),
        "best_epoch": control.get("best_epoch"),
        "best_iteration": control.get("best_iteration"),
        "development_engine_ids": sorted(dev_ids),
        "calibration_engine_ids": sorted(cal_ids),
        "development_engine_ids_sha256": canonical_sha256_json(sorted(dev_ids)),
        "calibration_engine_ids_sha256": canonical_sha256_json(sorted(cal_ids)),
        "python": sys.version.split()[0],
        "software_versions": versions,
    }
    if extra:
        meta.update(extra)
    return meta


def run_cv_fold(candidate: dict, fold: dict, frame: pd.DataFrame, cal_ids: set[int],
                data_dir: str | Path, splits_dir: str | Path, *, epochs: int = 60,
                batch_size: int = 256, patience: int = 8, seed: int = SEED) -> dict:
    """Two-stage nested CV for one candidate-fold; outer evaluation is untouched."""
    window = candidate["window"]
    outer_train, outer_eval = fold["outer_train"], fold["outer_eval"]
    inner_fit, inner_stop = inner_early_stop_split(outer_train, fold["fold"])

    # ---- M3 outer-fold falsification gates (M3_REPAIR_PLAN.md, required;
    #      explicit exceptions so they survive python -O) ----
    if not outer_train.isdisjoint(outer_eval):
        raise ValueError(f"fold {fold['fold']} leakage gate: outer train/eval overlap "
                         f"({sorted(outer_train & outer_eval)})")
    if not inner_fit.isdisjoint(inner_stop):
        raise ValueError(f"fold {fold['fold']} leakage gate: inner fit/stop overlap "
                         f"({sorted(inner_fit & inner_stop)})")
    if inner_fit | inner_stop != outer_train:
        raise ValueError(f"fold {fold['fold']} leakage gate: inner split must cover outer "
                         f"train (missing {sorted(outer_train - (inner_fit | inner_stop))}, "
                         f"extra {sorted((inner_fit | inner_stop) - outer_train)})")
    if not cal_ids.isdisjoint(outer_train):
        raise ValueError(f"fold {fold['fold']} leakage gate: calibration engines in outer "
                         f"training ({sorted(cal_ids & outer_train)})")
    if not cal_ids.isdisjoint(outer_eval):
        raise ValueError(f"fold {fold['fold']} leakage gate: calibration engines in outer "
                         f"evaluation ({sorted(cal_ids & outer_eval)})")
    if not (outer_eval.isdisjoint(inner_fit) and outer_eval.isdisjoint(inner_stop)):
        raise ValueError(f"fold {fold['fold']} leakage gate: outer eval engines leak into "
                         f"inner splits ({sorted(outer_eval & (inner_fit | inner_stop))})")

    manifest = load_manifest(Path(splits_dir) /
                             f"fd001_m3_outer_fold{fold['fold']}_cutoffs.csv")
    if not (len(manifest) == len(outer_eval) * 5 == 85):
        raise ValueError(
            f"fold {fold['fold']} completeness gate: outer-fold cutoff manifest must have "
            f"len(outer_eval) * 5 == 85 rows, got {len(manifest)} "
            f"(len(outer_eval)={len(outer_eval)})")
    trajectories = {
        int(e): g for e, g in frame[frame["engine_id"].isin(outer_eval)].groupby("engine_id")
    }

    start = time.perf_counter()
    if candidate["model"] == "rf":
        model, scaler, parameters, feature_count = _stage2_fit(
            candidate, frame, outer_train, {}, fold["fold"], batch_size=batch_size, seed=seed)
        control = {"best_epoch": None, "best_iteration": None,
                   "inner_seed": None, "stage": "single (rf: no early stopping)"}
        notes = "variable-history features (observed cycles up to cutoff only); no early stopping"
    elif candidate["model"] == "xgboost":
        best_iteration, control = _stage1_xgboost(
            candidate, frame, inner_fit, inner_stop, fold["fold"], seed=seed)
        control["stage"] = "stage1 xgb inner-stop -> stage2 fixed n_estimators"
        model, scaler, parameters, feature_count = _stage2_fit(
            candidate, frame, outer_train, control, fold["fold"], seed=seed)
        notes = "variable-history features; fixed n_estimators = best_iteration + 1; outer eval untouched"
    else:
        best_epoch, control = _stage1_deep(
            candidate, frame, inner_fit, inner_stop, fold["fold"],
            epochs=epochs, batch_size=batch_size, patience=patience, seed=seed)
        control["stage"] = "stage1 inner-stop early stop -> stage2 fixed epochs"
        model, scaler, parameters, feature_count = _stage2_fit(
            candidate, frame, outer_train, control, fold["fold"],
            batch_size=batch_size, seed=seed)
        notes = ("padded+masked training sequences; fixed-duration retrain, "
                 "NO validation data; outer eval untouched")
    training_time = round(time.perf_counter() - start, 2)

    pred = evaluate_manifest(manifest, trajectories,
                             make_predictor(candidate["model"], model, scaler,
                                            window=window))
    y_true = manifest["true_raw_rul"].to_numpy()
    engine_rows = engine_level_metrics(manifest, pred, candidate["id"], fold["fold"])
    row = {
        "candidate_id": candidate["id"],
        "fold": fold["fold"],
        "model": candidate["model"],
        "window": window,
        "outer_train_engine_count": len(outer_train),
        "outer_eval_engine_count": len(outer_eval),
        "inner_fit_engine_count": len(inner_fit),
        "inner_stop_engine_count": len(inner_stop),
        "inner_seed": control.get("inner_seed"),
        "best_epoch": control.get("best_epoch"),
        "best_iteration": control.get("best_iteration"),
        "validation_sample_count": len(manifest),
        "RMSE": round(float(rmse(y_true, pred)), 4),
        "MAE": round(float(mae(y_true, pred)), 4),
        "R2": round(float(r2(y_true, pred)), 4),
        "NASA_total": round(float(nasa_score(y_true, pred)), 2),
        "NASA_mean_per_engine": round(float(np.mean([e["NASA_sum"] for e in engine_rows])), 4),
        "signed_bias_mean": round(float(np.mean(pred - y_true)), 4),
        "training_time": training_time,
        "parameters": parameters,
        "notes": notes,
        "feature_count": feature_count,
    }
    return {"row": row, "engine_rows": engine_rows, "prediction_rows": [
        {"candidate_id": candidate["id"], "fold": fold["fold"],
         "engine_id": int(r.engine_id), "cutoff_cycle": int(r.cutoff_cycle),
         "true_raw_rul": float(r.true_raw_rul), "prediction": float(p)}
        for r, p in zip(manifest.itertuples(index=False), pred)],
        "control": control, "metadata": run_metadata(
            "FD001", candidate["id"], fold["fold"], control, outer_train | outer_eval,
            cal_ids, window)}


def engine_level_metrics(manifest: pd.DataFrame, pred: np.ndarray,
                         candidate_id: str, fold: int) -> list[dict]:
    """Per-engine metrics over its 5 checkpoint rows (macro-average unit = engine)."""
    out = []
    for engine, g in manifest.assign(prediction=pred).groupby("engine_id"):
        y = g["true_raw_rul"].to_numpy()
        p = g["prediction"].to_numpy()
        out.append({
            "candidate_id": candidate_id,
            "fold": fold,
            "engine_id": int(engine),
            "n_checkpoints": int(len(g)),
            "RMSE": float(rmse(y, p)),
            "MAE": float(mae(y, p)),
            "NASA_sum": float(nasa_score(y, p)),
            "signed_bias_mean": float(np.mean(p - y)),
        })
    return out


def assert_cv_complete(fold_rows: list[dict], candidates: list[dict] | None = None) -> int:
    """Hard completeness gate: every declared candidate must have folds {1..5}.

    Raises ValueError (never a bare assert, survives python -O) on incompleteness —
    this is an external-data falsification gate.
    """
    candidates = candidates or CV_CANDIDATES
    df = pd.DataFrame(fold_rows)
    for cid in [c["id"] for c in candidates]:
        folds = sorted(int(f) for f in df[df.candidate_id == cid]["fold"].unique())
        if folds != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"candidate {cid} incomplete: folds {folds} (requires [1, 2, 3, 4, 5])")
        if len(df[df.candidate_id == cid]) != 5:
            raise ValueError(
                f"candidate {cid} incomplete: {len(df[df.candidate_id == cid])} fold rows "
                f"(requires exactly 5)")
    n = len(df)
    if n != len(candidates) * 5:
        raise ValueError(
            f"CV completeness gate failed: {n} candidate-fold rows "
            f"(requires {len(candidates) * 5})")
    return n


def cv_summary(fold_rows: list[dict], candidates: list[dict] | None = None) -> list[dict]:
    """Macro metrics across outer folds for every candidate (policy input)."""
    candidates = candidates or CV_CANDIDATES
    assert_cv_complete(fold_rows, candidates)
    df = pd.DataFrame(fold_rows)
    out = []
    for cid, g in df.groupby("candidate_id"):
        base = {k: g.iloc[0][k] for k in ("candidate_id", "model", "window",
                                          "parameters", "notes")}
        for metric in ("RMSE", "MAE", "R2", "NASA_total", "NASA_mean_per_engine",
                       "signed_bias_mean", "training_time"):
            base[f"{metric}_mean"] = round(float(g[metric].mean()), 4)
            base[f"{metric}_std"] = round(float(g[metric].std(ddof=1)), 4)
        out.append(base)
    return out


def apply_selection_policy(summary: list[dict], n_folds: int = 5) -> dict:
    """Pre-declared M3 selection rule (see M3_REPAIR_PLAN.md).

    PRIMARY: lowest mean NASA per engine. GUARDRAIL: candidates whose NASA mean
    is within one pooled standard error SE = sqrt((s1^2 + s2^2)/n_folds) of the
    best are tied; prefer lowest RMSE among the tie. TIE: smallest |signed bias|.
    """
    df = pd.DataFrame(summary).sort_values("NASA_mean_per_engine_mean").reset_index(drop=True)
    best = df.iloc[0]
    tied = [best]
    for _, cand in df.iloc[1:].iterrows():
        se = float(np.sqrt((best["NASA_mean_per_engine_std"] ** 2 +
                            cand["NASA_mean_per_engine_std"] ** 2) / n_folds))
        if abs(best["NASA_mean_per_engine_mean"] - cand["NASA_mean_per_engine_mean"]) < se:
            tied.append(cand)
    tie_df = pd.DataFrame(tied).copy()
    tie_df["abs_signed_bias_mean_mean"] = tie_df["signed_bias_mean_mean"].abs()
    winner = tie_df.sort_values(["RMSE_mean", "abs_signed_bias_mean_mean"]).iloc[0]
    accuracy = df.sort_values("RMSE_mean").iloc[0]
    return {
        "methodology": "m3",
        "rule": ("PRIMARY lowest mean NASA per engine; guardrail: within one pooled "
                 "standard error prefer lower RMSE; tie: smaller |signed bias|"),
        "pooled_se_ties": [str(c["candidate_id"]) for c in tied],
        "accuracy_champion": str(accuracy["candidate_id"]),
        "nasa_risk_champion": str(winner["candidate_id"]),
        "deployment_selection": str(winner["candidate_id"]),
        "accuracy_champion_RMSE_mean": float(accuracy["RMSE_mean"]),
        "deployment_selection_metrics": {
            k: float(winner[k]) for k in (
                "RMSE_mean", "RMSE_std", "MAE_mean", "R2_mean",
                "NASA_mean_per_engine_mean", "NASA_mean_per_engine_std",
                "signed_bias_mean_mean", "signed_bias_mean_std")
        },
    }


def final_duration_rule(best_epochs_csv: str | Path, candidate_id: str) -> dict:
    """Deterministic aggregate of development-only best epochs (M3 rule)."""
    df = pd.read_csv(best_epochs_csv)
    rows = df[df.candidate_id == candidate_id]
    # external-data completeness gate: explicit exception (survives python -O)
    if len(rows) != 5:
        raise ValueError(
            f"{candidate_id} needs 5 best-epoch rows, got {len(rows)} in {best_epochs_csv}")
    if rows.iloc[0]["best_iteration"] is not None and pd.notna(rows.iloc[0]["best_iteration"]):
        n = round(float(np.median(rows["best_iteration"].to_numpy(dtype=float)))) + 1
        return {"rule": "n_estimators = round(median(best_iteration)) + 1",
                "n_estimators": int(n), "best_iterations": [int(x) for x in rows["best_iteration"]]}
    epochs = int(round(float(np.median(rows["best_epoch"].to_numpy(dtype=float)))))
    return {"rule": "final_epoch_count = round(median(best_epoch))",
            "epochs": epochs, "best_epochs": [int(x) for x in rows["best_epoch"]]}