> **SUPERSEDED BY METHODOLOGY V2.1** — this report documents the V2
> methodology as historical record. V2.1 corrects the lifetime semantics
> (observed history vs implied failure cycle), replaces the single 15-engine
> validation with 5-fold engine-group CV, recalibrates conformal at the
> engine level, and fixes the FD004 collapse with per-regime scaling.
> See V2_1_REPAIR_PLAN.md and reports/v2_1_methodology.md.

# Phase V2-6 — Error Analysis of the Frozen Model

Date: 2026-08-15  |  Model: frozen GRU w45 huber (`models/v2_frozen_gru_w45_huber.keras`)  |  Target: raw RUL

Two samples, same frozen model: the 75-row validation manifest (model-selection sample) and the 100-engine official FD001 test (**post-hoc** — labels inspected in the V2-0 audit). Per-engine rows: `experiments/v2_error_profile.csv` (gitignored); reproducible via `scripts/analyze_v2_errors.py`. No training, no calibration-engine contact.

## Headline comparison

| | validation (n=75) | official (n=100) |
|---|---|---|
| RMSE | 13.74 | 29.04 |
| MAE | 9.69 | 19.17 |
| R² | 0.877 | 0.512 |
| NASA | 200.01 | 77,387.53 |
| bias (mean / median) | −2.04 / −1.40 | **+8.89 / +1.65** |
| \|err\| < 13 cycles | 74.7% | 54.0% |
| late / early | 31 / 44 | 57 / 43 |
| \|err\| > 50 late / early | 0 / 0 | **11 / 0** |
| corr(true RUL, err) | −0.301 | +0.180 |
| corr(lifetime, err) | −0.622 | −0.604 |

The frozen model is nearly unbiased on the validation manifest and mildly **conservative (early)** at high RUL there, but **systematically late** on the official test, with 11 engines missed late by >50 cycles and none missed early by >50.

## Error vs true RUL (buckets)

Validation:

| true RUL | n | mean err | RMSE | MAE | late% | \|err\|<13% |
|---|---|---|---|---|---|---|
| [0, 20) | 17 | −1.21 | 4.24 | 3.42 | 29% | 100% |
| [20, 40) | 16 | +1.21 | 6.57 | 5.44 | 50% | 94% |
| [40, 60) | 13 | −0.09 | 10.29 | 7.93 | 38% | 77% |
| [60, 80) | 11 | −0.56 | 10.68 | 8.65 | 45% | 91% |
| [80, 100) | 5 | −1.74 | 19.27 | 18.01 | 60% | 20% |
| [100, 125) | 10 | −3.73 | 22.06 | 19.51 | 50% | 30% |
| [125, 160) | 2 | −25.77 | 28.59 | 25.77 | 0% | 0% |
| [160, 200) | 1 | −46.61 | 46.61 | 46.61 | 0% | 0% |

Official:

| true RUL | n | mean err | RMSE | MAE | late% | \|err\|<13% |
|---|---|---|---|---|---|---|
| [0, 20) | 13 | +0.67 | 5.10 | 3.47 | 38% | 92% |
| [20, 40) | 15 | +1.25 | 5.12 | 3.98 | 60% | 100% |
| [40, 60) | 11 | +4.18 | 12.90 | 7.64 | 55% | 91% |
| [60, 80) | 6 | **+25.39** | 48.85 | 32.79 | 67% | 33% |
| [80, 100) | 21 | +6.23 | 31.29 | 22.79 | 52% | 48% |
| [100, 125) | 23 | **+20.29** | 38.89 | 31.78 | 70% | 22% |
| [125, 150) | 11 | +5.99 | 33.14 | 29.27 | 55% | 0% |

On the manifest the model degrades gracefully and *early* at high RUL (never dangerous). On the official test the trouble is concentrated at **true RUL 60–128**: 40 of 100 engines sit there and they are missed late by ~6–25 cycles on average, up to +110.6.

## The worst engines (official test)

| engine | lifetime | true RUL | pred | err | NASA pts |
|---|---|---|---|---|---|
| 67 | 71 | 77 | 187.6 | +110.6 | 63,833 |
| 78 | 72 | 107 | 196.2 | +89.2 | 7,508 |
| 6 | 105 | 93 | 164.6 | +71.6 | 1,292 |
| 15 | 76 | 83 | 154.3 | +71.3 | 1,249 |
| 65 | 71 | 128 | 195.7 | +67.7 | 867 |

All five failed **young** (lifetime 71–105 cycles; train engines average ~206) with moderate true RUL (77–128). The model reads their sensors as healthy and predicts a long remaining life. NASA on the official test is 99.8% late-miss driven, of which **99.1% comes from the 11 engines missed late by >50 cycles** (5 engines alone = 96.6%, 74,749 pts of 77,388).

## Padded engines (lifetime < 45 cycles, official test)

All 4 are missed **late**: +35.4, +48.2, +60.3, +66.3. Short-observed-history engines are systematically overpredicted — the padding representation itself is not the cause (training includes padded sequences), but short-lifetime engines are rare in training (mean train lifetime ≈ 206) and the model has not learned them.

## Distribution shift (the structural driver)

| sample | RUL mean | RUL median | RUL max |
|---|---|---|---|
| train sequences (14,235) | 106.7 | 101 | 361 |
| validation manifest (75 rows) | 53.0 | 42 | 171 |
| official test (100 engines) | 75.5 | 86 | 145 |

The validation manifest samples cutoffs at fractions 0.50–0.95 of each engine's lifetime, so its RUL values cluster **low** (median 42) — exactly the range where the model is accurate and unbiased. The official test's RUL distribution is higher (median 86) and sits in the region where the model overpredicts. The consistent `corr(lifetime, err) ≈ −0.6` on both samples is the underlying signature: the model has learned "engines live long" and underpredicts on long-lifetime engines (conservative, safe) while overpredicting on short-lifetime engines (dangerous, late).

### The sharpest split: lifetime below the training minimum

Train engine lifetimes: mean 206, median 199, **min 128**. Official engines failing before 128 cycles have **never been seen in training**:

| official engines | n | missed late | mean err | NASA share |
|---|---|---|---|---|
| lifetime < 128 (below train min) | 44 | 35 (80%) | **+25.7** | **99.8%** |
| lifetime ≥ 128 (inside train range) | 56 | 22 (39%) | −4.3 | 0.2% |

Within the trained lifetime range the model is mildly conservative (safe). The entire NASA tail is produced by engines outside the training envelope — a domain-shift failure, not a noise failure. This also explains the 4 padded engines (lifetime < 45): all are below the train minimum and all are overpredicted.

## Implications

1. **RMSE/MAE/R² are not the risk metric here.** The operational risk is entirely in the late tail: 99.8% of official NASA score comes from engines with lifetime below the training minimum (128 cycles), missed late by >50 cycles on average.
2. **Domain-shift-aware deployment:** any serving system should flag engines whose observed history is short (lifetime ≲ 130 cycles) as out-of-distribution and refuse/hedge the raw prediction — they are exactly the engines the model cannot see.
3. **V2-8 (conformal calibration) should target the lower prediction bound** — the early-warning side — and be assessed with an asymmetric/conservative score, not just interval coverage.
4. A conservative bias correction (shift predictions down) is the cheapest mitigation for the frozen model; its cost is a small RMSE loss, its gain is collapsing the NASA tail. Quantify before adopting (V2-8).
5. Young-failing engines (lifetime ≲ 110 cycles) deserve feature-level scrutiny in V2-7 (SHAP) — they carry most of the risk.

Next: **Phase V2-7 — explainability (SHAP)** on the frozen model (currently shap is not installed; the phase will add it to the venv and requirements).