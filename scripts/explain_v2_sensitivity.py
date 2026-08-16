"""Phase V2-7: explainability of the frozen model via sensor-level attribution.

Method (sensitivity/occlusion, NOT SHAP): for each sensor s, replace sensor s's
values across the whole window with the per-sensor background median (from
full-length train windows) and measure the prediction drop:

    attribution_s = f(window) - f(window with sensor s replaced by median)

This is the marginal contribution of sensor s relative to the background
distribution. (The script was renamed from explain_v2_shap.py in V2.1 because
it is not SHAP; keras 3 breaks shap's deep/kernel explainers on this model —
see reports/v2_explainability.md.) Attributions are in raw-RUL units
(prediction units). Sum-check: sum_s attribution_s vs f(window) -
f(window with ALL sensors replaced); gap reported honestly (nonlinear model).

Target groups (from experiments/v2_error_profile.csv, full windows only):
  late-miss  : official engines with error > +50 cycles (the NASA tail)
  accurate   : official engines with |err| < 13 cycles
  engine 67  : the worst single engine (+110.6 cycles) — also position-level
               attribution (replace one window row at a time).

Writes experiments/v2_shap_attributions.csv and
experiments/v2_shap_position_engine67.csv (gitignored). No training, no
calibration contact.
"""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, models as keras_models

from rul_prediction.benchmark.v2 import ROOT, load_v2_artifacts
from rul_prediction.data.loader import SENSOR_COLUMNS, load_test
from rul_prediction.data.preprocessing import transform
from rul_prediction.data.v2_preprocessing import build_v2_train_sequences

WINDOW = 45
N_SENSORS = len(SENSOR_COLUMNS)
MODEL_PATH = ROOT / "models" / "v2_frozen_gru_w45_huber.keras"
PROFILE_CSV = ROOT / "experiments" / "v2_error_profile.csv"
OUT_SENSOR = ROOT / "experiments" / "v2_shap_attributions.csv"
OUT_POSITION = ROOT / "experiments" / "v2_shap_position_engine67.csv"


def load_windows() -> tuple[np.ndarray, dict]:
    """Full (unpadded) scaled windows: train (background) + official test by engine."""
    a = load_v2_artifacts()
    frame, scaler = a["frame"], a["scaler"]
    train_rows = frame[frame["engine_id"].isin(a["train_ids"])]
    X, y, ids, n_obs, masks = build_v2_train_sequences(
        transform(train_rows, SENSOR_COLUMNS, scaler),
        train_rows["engine_id"].to_numpy(), np.zeros(len(train_rows), np.float32), WINDOW)
    full = X[n_obs == WINDOW]
    test = load_test("FD001")
    windows = {}
    for engine, g in test.groupby("engine_id"):
        g = g.sort_values("cycle")
        if len(g) < WINDOW:
            continue
        scaled = scaler.transform(g[SENSOR_COLUMNS].to_numpy(dtype=float))
        windows[int(engine)] = scaled[-WINDOW:].astype(np.float32)
    return full, windows


def main() -> None:
    full_train, test_windows = load_windows()
    model = keras_models.load_model(MODEL_PATH)
    inp = Input(shape=(WINDOW, N_SENSORS))
    mask = layers.Lambda(lambda t: tf.ones((tf.shape(t)[0], WINDOW), dtype="float32"))(inp)
    wrapped = Model(inp, model([inp, mask]))

    def predict(x: np.ndarray) -> np.ndarray:
        return wrapped.predict(np.asarray(x, dtype=np.float32), verbose=0)[:, 0]

    median = np.median(full_train, axis=(0, 1))  # per-sensor background median
    median_row = np.median(full_train, axis=0)   # per-position background median row

    profile = pd.read_csv(PROFILE_CSV)
    off = profile[profile["sample"] == "official"].set_index("engine_id")
    late_ids = sorted(off.index[(off["error"] > 50) & (off["lifetime_cycles"] >= WINDOW)])
    acc_ids = sorted(off.index[(off["error"].abs() < 13) & (off["lifetime_cycles"] >= WINDOW)])
    assert late_ids and acc_ids and 67 in test_windows, "target groups must be non-empty"

    def attributions(win: np.ndarray) -> tuple[np.ndarray, float]:
        """Per-sensor marginal contributions + sum-check (full-background prediction)."""
        base = float(predict(win[None])[0])
        variants = np.repeat(win[None], N_SENSORS, axis=0)
        for s in range(N_SENSORS):
            variants[s, :, s] = median[s]
        delta = base - predict(variants)
        all_med = np.full_like(win, median)
        gap = abs(float(delta.sum()) - (base - float(predict(all_med[None])[0])))
        return delta, gap

    groups = {"late-miss": late_ids, "accurate": acc_ids}
    agg = []
    for name, ids in groups.items():
        deltas = np.stack([attributions(test_windows[e])[0] for e in ids])
        gaps = np.array([attributions(test_windows[e])[1] for e in ids])
        preds = np.array([float(predict(test_windows[e][None])[0]) for e in ids])
        truth = np.array([off.loc[e, "true_rul"] for e in ids])
        errors = preds - truth
        for s in range(N_SENSORS):
            agg.append({"group": name, "engine_count": len(ids),
                        "sensor": SENSOR_COLUMNS[s],
                        "mean_attribution": float(deltas[:, s].mean()),
                        "mean_abs_attribution": float(np.abs(deltas[:, s]).mean()),
                        "std_attribution": float(deltas[:, s].std())})
        print(f"[{name}] n={len(ids)} engines; "
              f"mean pred {preds.mean():.1f}  mean true {truth.mean():.1f}  mean err {errors.mean():+.1f}")
        print(f"  sum-check |sum(delta) - (pred - pred(all-median))| mean {gaps.mean():.2f}")
        top = np.argsort(np.abs(deltas.mean(axis=0)))[-5:][::-1]
        for s in top:
            print(f"  {SENSOR_COLUMNS[s]:>9}: mean attr {deltas[:, s].mean():+7.2f}  "
                  f"mean |attr| {np.abs(deltas[:, s]).mean():6.2f}")

    eng67 = test_windows[67]
    d67, gap67 = attributions(eng67)
    pred67 = float(predict(eng67[None])[0])
    print(f"\nengine 67: pred {pred67:.1f} (true {off.loc[67, 'true_rul']:.1f})  "
          f"sum(delta) {d67.sum():.1f}  sum-check gap {gap67:.2f}")
    for s in np.argsort(np.abs(d67))[::-1][:6]:
        print(f"  {SENSOR_COLUMNS[s]:>9}: {d67[s]:+7.2f}")

    base_row = float(predict(eng67[None])[0])
    rows = np.repeat(eng67[None], WINDOW, axis=0)
    for p in range(WINDOW):
        rows[p, p, :] = median_row[p]
    pos_delta = base_row - predict(rows)
    print("\nengine 67 position-level attribution (recent=end of window):")
    for p in range(WINDOW - 1, -1, -5):
        print(f"  position {p + 1:>2}: {pos_delta[p]:+7.2f}")
    print(f"  sum(pos_delta) {pos_delta.sum():.1f} vs pred gap {base_row - pred67:.1f}")

    with OUT_SENSOR.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(agg[0].keys()))
        writer.writeheader()
        writer.writerows(agg)
    with OUT_POSITION.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["position", "attribution"])
        writer.writeheader()
        for p in range(WINDOW):
            writer.writerow({"position": p + 1, "attribution": float(pos_delta[p])})
    print(f"wrote -> {OUT_SENSOR}")
    print(f"wrote -> {OUT_POSITION}")


if __name__ == "__main__":
    main()