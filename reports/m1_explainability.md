> **SUPERSEDED BY METHODOLOGY M2** — this report documents the M1
> methodology as historical record. M2 corrects the lifetime semantics
> (observed history vs implied failure cycle), replaces the single 15-engine
> validation with 5-fold engine-group CV, recalibrates conformal at the
> engine level, and fixes the FD004 collapse with per-regime scaling.
> See M2_REPAIR_PLAN.md and reports/m2_methodology.md.

# Phase M1-7 — Explainability of the Frozen Model

Date: 2026-08-15  |  Model: frozen GRU w45 huber (`models/m1_frozen_gru_w45_huber.keras`)  |  Target: raw RUL

## Method

Keras 3.15 breaks all three classic shap explainers on this model (Deep/Gradient/Kernel — `Functional.call` / rank errors; verified by probe). The library constraint is resolved by a **sensor-level counterfactual attribution**, computed exactly:

> `attribution_s = f(window) − f(window with sensor s replaced by its training median)`

i.e. "how many raw-RUL units would the prediction drop if sensor s had been typical". Units are prediction units (cycles). Evaluated on the 45-cycle terminal window of each official-test engine (full windows only; the 4 padded engines are excluded — their failures were analyzed in M1-6). Background median: per-sensor median over all 14,235 full-length train windows. **This is not an additive Shapley decomposition**: single-sensor replacements double-count shared signal, and the measured sum-check gap (|Σ attributions − total replacement effect|) is 74–109 cycles — reported honestly, not hidden. Attributions are directionally meaningful per sensor; they must not be summed.

`shap` 0.52.0 is now installed (it was declared in `requirements.txt` but missing — M1-0 audit finding; no requirement change needed). Repro: `scripts/explain_m1_sensitivity.py`; per-sensor CSV in `experiments/m1_sensitivity_attributions.csv`, engine-67 position profile in `experiments/m1_sensitivity_position_engine67.csv`.

## Groups (official test, from `experiments/m1_error_profile.csv`)

| group | n | mean pred | mean true | mean err |
|---|---|---|---|---|
| late-miss (err > +50, full windows) | 9 | 171.1 | 102.2 | +68.8 |
| accurate (\|err\| < 13, full windows) | 54 | 49.8 | 49.0 | +0.9 |

(2 of the 11 late>50 engines are padded — excluded here, covered in M1-6.)

## Per-sensor attribution: late-miss vs accurate

| sensor | late-miss mean attr | accurate mean attr | contrast (LM − Acc) |
|---|---|---|---|
| **sensor_7** | **+21.7** | −5.2 | **+26.9** |
| **sensor_2** | **+18.6** | −3.1 | **+21.7** |
| **sensor_8** | **+13.1** | −6.8 | **+19.9** |
| **sensor_6** | **+18.2** | −0.0 | **+18.2** |
| **sensor_4** | **+11.2** | −7.2 | **+18.4** |
| sensor_12 | −6.2 | −8.7 | +2.4 |
| sensor_11 | −2.3 | −6.4 | +4.2 |
| sensor_20 | −9.6 | −9.1 | −0.5 |
| sensors 1, 5, 10, 16, 18, 19 | ≈ 0 | ≈ 0 | 0 |

**Internal consistency check:** the six sensors known to be constant in FD001 (1, 5, 10, 16, 18, 19) have zero attribution (≤ 2e-5) — the model ignores them, exactly as it should.

**Finding — sign flip on the same sensors.** In accurate engines, sensors 2, 4, 7, 8 *suppress* the prediction (−3 to −7 cycles each): the model reads their deviations from typical as degradation. In late-miss engines the same sensors *inflate* it (+11 to +22): the young-failing engines present these sensors as healthy, and the model trusts them — the machine-level cause of the +68.8-cycle mean overprediction. Sensor 6 is one-sided (late-miss +18.2, accurate ≈0): its values look healthy only in the failing group.

## The worst engine (67, err +110.6)

| sensor | attribution | | window position (1 = oldest) | attribution |
|---|---|---|---|---|
| sensor_2 | +47.7 | | 40 (recent 5 cycles) | +34.8 |
| sensor_7 | +42.0 | | 30 (recent 15 cycles) | +22.8 |
| sensor_4 | +18.0 | | 45 (current) | +19.6 |
| sensor_8 | +16.1 | | 35 | +15.6 |
| sensor_6 | +13.8 | | 25 | +9.3 |
| sensor_14 | +8.5 | | 1–20 (older history) | −2.9 … −6.9 |

Engine 67's overprediction is driven by the **recent 15 cycles** (positions 30–45, +15 to +35 each) and by the same sensor set (2, 4, 6, 7, 8). Its early history mildly suppresses the prediction; its recent window looks healthy to the model. This is the sensor-level fingerprint of the M1-6 finding: engines that fail young look healthy in the recent window, and the model has no training experience (lifetime < 128) to override the healthy reading.

## Interpretation for the pipeline

1. **Sensors 2, 4, 6, 7, 8 are the risk-relevant signal.** The model's late misses are exactly the engines where these sensors present healthy values despite imminent failure.
2. **M1-8 (conformal) has a concrete target:** the interval must widen when the recent-window profile of these sensors is "healthy-looking" on a short-lifetime engine — the counterfactual sensitivity above is the natural input for an OOD/hedging rule.
3. The attribution method (exact counterfactual sensitivity) is a documented, reproducible substitute for shap on keras 3 — the non-additivity caveat (sum-check gap 74–109) is measured and disclosed.

Next: **Phase M1-8 — conformal/uncertainty calibration** on the 75 calibration-manifest engines (untouched so far), targeting a conservative lower bound for raw RUL.