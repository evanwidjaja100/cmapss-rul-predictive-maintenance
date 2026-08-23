> **SUPERSEDED BY METHODOLOGY M2** — this report documents the M1
> methodology as historical record. M2 corrects the lifetime semantics
> (observed history vs implied failure cycle), replaces the single 15-engine
> validation with 5-fold engine-group CV, recalibrates conformal at the
> engine level, and fixes the FD004 collapse with per-regime scaling.
> See M2_REPAIR_PLAN.md and reports/m2_methodology.md.

# Phase M1-8 — Conformal Uncertainty Calibration

Date: 2026-08-15  |  Model: frozen GRU w45 huber  |  Target: raw RUL  |  Calibration set: the **75 fixed calibration-manifest rows** (15 engines × 5 lifecycle fractions) — untouched by every prior phase; first use here.

## Method

Split (inductive) conformal prediction, exactly as implemented in `src/rul_prediction/evaluation/conformal.py`:

- nonconformity scores = absolute residuals of the frozen model on the 75 calibration rows;
- `q = conformal_quantile(scores, α)` = ⌈(n+1)(1−α)⌉-th smallest score → interval ŷ ± q with marginal coverage **≥ 1−α on exchangeable data**;
- the **early-warning value ŷ − q** is also evaluated as a prediction (the conservative-bias tradeoff identified in M1-6).

Calibration residuals: n=75, mean 7.87, median 4.25, max 51.27.

## Results

| α | target cov | q (cycles) | mean width | cal cov | val cov | off cov |
|---|---|---|---|---|---|---|
| 0.1 | 90% | 24.10 | 48.2 | 0.920 | **0.880** | **0.690** |
| 0.2 | 80% | 11.77 | 23.5 | 0.813 | 0.693 | 0.520 |
| 0.3 | 70% | 7.53 | 15.1 | 0.720 | 0.573 | 0.440 |

Calibration coverage meets the finite-sample guarantee by construction (92 ≥ 90, 81 ≥ 80, 72 ≥ 70).

### Coverage falls below nominal on held-out samples — measured, not assumed

1. **Validation (75 rows): 88/69/57%.** The calibration engines are *easier* for the model than the validation engines (calibration median residual 4.25 vs validation MAE 9.69) — different engine draws with the same lifecycle fractions are not perfectly exchangeable with each other. For α=0.1 the shortfall is 66/75 vs 67.5 expected, inside ~0.6σ of nominal; the pattern persists across all α.
2. **Official test (100 engines): 69/52/44%.** Exchangeability is violated by distribution shift. The split is sharp: engines with lifetime ≥ 128 (inside the training range) achieve **85.7%** coverage at α=0.1 — close to nominal — while short-lifetime engines (lifetime < 128, below the training minimum) get **47.7%**. Conformal is calibrated exactly where the model is trained, and uncalibrated exactly where the M1-6 risk lives.

### The conservative lower bound (ŷ − q) as an early-warning prediction

| α (q) | validation RMSE / NASA | official RMSE / NASA | plain RMSE / NASA |
|---|---|---|---|
| 0.1 (24.1) | 29.46 / 1075.4 | 31.55 / **8184.6** | 13.74 / 200.0 (val), 29.04 / 77387.5 (off) |
| 0.2 (11.8) | 19.37 / 388.6 | **27.79** / 24225.1 | |
| 0.3 (7.5) | 16.62 / 279.3 | 27.68 / 36661.0 | |

- **On the validation manifest (near-zero bias):** any downward shift hurts (NASA 200 → 279–1075). The plain prediction is right there.
- **On the official test (late-biased):** the shift *corrects* the bias — at α=0.2 the official RMSE improves (27.79 < 29.04) while NASA improves 3.2× (24,225 vs 77,388); at α=0.1 NASA improves 9.5× (8,185) at RMSE cost 31.6 vs 29.0.
- **Honesty note:** choosing α by official-test outcomes is post-hoc tuning; the in-sample default remains α=0.1 (nominal 90%). The observation stands regardless: the official late bias makes a conservative shift systematically beneficial there.
- **The shift cannot fix the OOD tail.** Engine 67 (the worst miss) remains **70–81% of the residual NASA** after every shift (ŷ−q of 175.8 vs true 77 still scores ~19,700 at α=0.2). A constant shift corrects the *bias*; it does not correct the *domain shift* — the complementary guard is the M1-6 finding: flag engines with lifetime ≲ 128 cycles (below training minimum) as out-of-distribution and hedge/refuse the raw prediction.

## Artifacts & reproducibility

- `reports/tables/m1_conformal_calibration.csv` — full per-α table (tracked).
- `experiments/m1_calibration_scores.csv`, `experiments/m1_conformal_intervals.csv` (α=0.1, lo/hi per engine) — gitignored.
- `scripts/calibrate_m1_conformal.py` — reproduces everything (asserts calibration coverage ≥ guarantee).
- `src/rul_prediction/evaluation/conformal.py` + `tests/test_m1_conformal.py` — 4 new tests covering the finite-sample quantile math, the by-construction calibration coverage, and empirical iid holdout coverage.

**Recommendation for M1-9 (serving/Streamlit):** expose intervals at α=0.1; show ŷ − q as the planning value; flag OOD engines (lifetime < 128 cycles) with the measured 47.7% coverage warning. Next: **Phase M1-9 — Streamlit serving app**.