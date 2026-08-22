"""Phase V2-6: residual/error analysis of the frozen V2 model.

Loads the frozen model (models/v2_frozen_gru_w45_huber.keras) and profiles
per-engine prediction errors on both evaluation samples:

* validation manifest  — 75 fixed rows (the model-selection sample)
* official FD001 test  — 100 engines, POST-HOC (labels inspected in V2-0 audit)

Outputs: experiments/v2_error_profile.csv (per-engine rows, gitignored) and a
printed summary used to author reports/v2_error_analysis.md. No training and
no calibration-engine contact.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from rul_prediction.benchmark.v2 import (ROOT, evaluate_official_test,
                                         load_v2_artifacts, make_predictor)
from rul_prediction.data.loader import load_rul, load_test
from rul_prediction.evaluation.manifest import evaluate_manifest
from rul_prediction.evaluation.metrics import mae, r2, rmse
from rul_prediction.evaluation.nasa_score import nasa_score

MODEL_NAME = "gru"
WINDOW = 45
MODEL_PATH = ROOT / "models" / "v2_frozen_gru_w45_huber.keras"
PROFILE_CSV = ROOT / "experiments" / "v2_error_profile.csv"


def nasa_points(y_true, y_pred):
    d = np.asarray(y_pred, float) - np.asarray(y_true, float)
    return np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)


def summarize(name, true, pred, extra_rows):
    d = pred - true
    pts = nasa_points(true, pred)
    total = float(pts.sum())
    rows = []
    for e, life, t, p in zip(extra_rows["engine_id"], extra_rows["lifetime_cycles"], true, pred):
        rows.append({"sample": name, "engine_id": int(e), "lifetime_cycles": int(life),
                     "true_rul": float(t), "prediction": float(p),
                     "error": float(p - t), "nasa_point": float(nasa_points(np.array([t]), np.array([p]))[0])})
    print(f"--- {name} (n={len(true)}) ---")
    print(f"RMSE {rmse(true, pred):.4f}  MAE {mae(true, pred):.4f}  R2 {r2(true, pred):.4f}  "
          f"NASA {total:.2f}")
    good = (np.abs(d) < 13).mean()
    late = (d > 0).sum()
    early = (d < 0).sum()
    print(f"bias: mean {d.mean():+.2f}  median {np.median(d):+.2f}")
    print(f"|err|<13: {good:.1%}   late: {late}   early: {early}   "
          f"late>50: {(d > 50).sum()}   early>50: {(d < -50).sum()}")
    return rows


def bucket_table(true, pred, edges):
    d = pred - true
    print(f"\nbucket (true RUL)      n    mean_err  rmse    mae   late%  |err|<13%")
    idx = np.digitize(true, edges)
    for b in range(len(edges) + 1):
        m = idx == b + 1
        if m.sum() == 0:
            continue
        lo = edges[b] if b < len(edges) else edges[-1]
        hi = edges[b + 1] if b + 1 < len(edges) else np.inf
        print(f"[{lo:>3},{hi:>4})  {m.sum():>4}  {d[m].mean():+8.2f}  {rmse(true[m], pred[m]):6.2f}  "
              f"{mae(true[m], pred[m]):6.2f}  {(d[m] > 0).mean():5.0%}  {(np.abs(d[m]) < 13).mean():7.0%}")


def main() -> None:
    artifacts = load_v2_artifacts()
    frame, scaler = artifacts["frame"], artifacts["scaler"]
    manifest, trajectories = artifacts["manifest"], artifacts["trajectories"]
    model = __import__("keras").models.load_model(MODEL_PATH)
    predictor = make_predictor(MODEL_NAME, model, scaler, window=WINDOW)

    val_true = manifest["true_raw_rul"].to_numpy()
    val_pred = evaluate_manifest(manifest, trajectories, predictor)
    val_life = {int(e): int(g["cycle"].max()) for e, g in frame.groupby("engine_id")}
    val_extra = {"engine_id": manifest["engine_id"].to_numpy(),
                 "lifetime_cycles": [val_life[int(e)] for e in manifest["engine_id"]]}

    official_metrics, official_rows = evaluate_official_test(MODEL_NAME, model, scaler, WINDOW)
    off_true = np.array([r["true_rul_official"] for r in official_rows], float)
    off_pred = np.array([r["prediction"] for r in official_rows], float)
    test = load_test("FD001")
    test_life = {int(e): int(g["cycle"].max()) for e, g in test.groupby("engine_id")}
    off_extra = {"engine_id": [r["engine_id"] for r in official_rows],
                 "lifetime_cycles": [test_life[int(r["engine_id"])] for r in official_rows]}

    rows = summarize("validation", val_true, val_pred, val_extra)
    rows += summarize("official", off_true, off_pred, off_extra)

    print("\n=== distribution of true RUL ===")
    print(f"train raw RUL (14,235 seqs): mean 106.7 median 101 max 361  (V2-2 stats)")
    print(f"validation manifest: mean {val_true.mean():.1f} median {np.median(val_true):.1f} "
          f"min {val_true.min()} max {val_true.max()}")
    print(f"official test: mean {off_true.mean():.1f} median {np.median(off_true):.1f} "
          f"min {off_true.min()} max {off_true.max()}")

    print("\n=== buckets: validation ===")
    bucket_table(val_true, val_pred, [0, 20, 40, 60, 80, 100, 125, 160, 200])
    print("\n=== buckets: official ===")
    bucket_table(off_true, off_pred, [0, 20, 40, 60, 80, 100, 125, 150])

    d_val, d_off = val_pred - val_true, off_pred - off_true
    print("\n=== correlations ===")
    print(f"validation: corr(true_rul, err) = {np.corrcoef(val_true, d_val)[0, 1]:+.3f}   "
          f"corr(lifetime, err) = {np.corrcoef(np.array(val_extra['lifetime_cycles'], float), d_val)[0, 1]:+.3f}")
    print(f"official:   corr(true_rul, err) = {np.corrcoef(off_true, d_off)[0, 1]:+.3f}   "
          f"corr(lifetime, err) = {np.corrcoef(np.array(off_extra['lifetime_cycles'], float), d_off)[0, 1]:+.3f}")

    print("\n=== worst 5 by |error| ===")
    for name, t, p, ex in (("validation", val_true, val_pred, val_extra),
                           ("official", off_true, off_pred, off_extra)):
        d = p - t
        o = np.argsort(np.abs(d))[-5:][::-1]
        print(name)
        for i in o:
            print(f"  engine {int(ex['engine_id'][i]):>3}  lifetime {ex['lifetime_cycles'][i]:>4}  "
                  f"true {t[i]:6.1f}  pred {p[i]:6.1f}  err {d[i]:+7.1f}  nasa {nasa_points(np.array([t[i]]), np.array([p[i]]))[0]:.1f}")

    print("\n=== worst 5 by NASA point ===")
    for name, t, p, ex in (("validation", val_true, val_pred, val_extra),
                           ("official", off_true, off_pred, off_extra)):
        d = p - t
        pts = nasa_points(t, p)
        o = np.argsort(pts)[-5:][::-1]
        print(name)
        for i in o:
            print(f"  engine {int(ex['engine_id'][i]):>3}  lifetime {ex['lifetime_cycles'][i]:>4}  "
                  f"true {t[i]:6.1f}  pred {p[i]:6.1f}  err {d[i]:+7.1f}  nasa {pts[i]:.1f}")

    print("\n=== NASA share by sign of error ===")
    for name, t, p in (("validation", val_true, val_pred), ("official", off_true, off_pred)):
        d = p - t
        pts = nasa_points(t, p)
        late_share = pts[d > 0].sum() / pts.sum()
        tail_share = pts[d > 50].sum() / pts.sum() if (d > 50).any() else 0.0
        print(f"{name}: late-miss share of NASA {late_share:.1%}   (>50-cycle late-miss share {tail_share:.1%})")

    print("\n=== padded engines (lifetime < 45) ===")
    for name, t, p, ex in (("validation", val_true, val_pred, val_extra),
                           ("official", off_true, off_pred, off_extra)):
        life = np.array([int(ex["lifetime_cycles"][i]) for i in range(len(t))])
        m = life < WINDOW
        if m.sum():
            print(f"{name}: {m.sum()} padded; errors: {np.round(p[m] - t[m], 1)}")
        else:
            print(f"{name}: no padded engines")

    with PROFILE_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote -> {PROFILE_CSV}")


if __name__ == "__main__":
    main()