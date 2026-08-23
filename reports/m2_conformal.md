# M2 Conformal Prediction (FD001, engine-cluster calibration)

Date: 2026-08-15 · Repair plan R9/R10/R11 · Supersedes the M1-8 row-level
conformal interval (q = 24.10, n = 75 rows) in `reports/m1_conformal.md`.

## Calibration unit = engine

The M1-8 calibration treated the 5 lifecycle checkpoints of each calibration
engine as independent rows (n = 75). They are not independent — they are
correlated residuals of the same engine — so the finite-sample guarantee was
mis-stated. M2 calibrates at the ENGINE level:

- for each of the 15 calibration engines, the nonconformity score is the max
  absolute error over its five checkpoints (0.25/0.45/0.65/0.80/0.95);
- `score_engine = max |prediction - true_raw_rul|` over the 5 cutoffs
  (`src/rul_prediction/evaluation/conformal.py::engine_cluster_scores`);
- quantile index `k = ceil((n+1)(1-alpha))` clamped to `1 <= k <= n`
  (`finite_sample_quantile_index`); with n = 15 and alpha = 0.1, k = 15.

**Guarantee (wording):** for exchangeable engines under the predefined
lifecycle-checkpoint scheme, the interval `pred +/- q` contains ALL FIVE
checkpoint RULs with probability >= 1 - alpha. Application to arbitrary
real-time uploaded trajectories is an engineering extrapolation, reported
separately from the formal guarantee.

## Calibration (frozen FD001 model, reports/tables/m2_conformal_calibration.csv)

Engine-level max-abs scores (n = 15): 10.9, 11.4, 13.6, 15.1, 15.8, 19.0,
23.5, 30.5, 31.8, 39.2, 41.8, 42.1, 48.1, 55.2, 70.3.

| alpha | k | q (cycles) |
|---|---|---|
| 0.10 | 15 | 70.34 |
| 0.20 | 13 | 48.11 |
| 0.30 | 12 | 42.13 |

Serving uses **alpha = 0.1, q = 70.34** (engine-cluster interval,
reports/tables/m2_conformal_q.csv).

## Held-out empirical coverage (reports/tables/m2_conformal_coverage.csv)

| alpha | q | official coverage | official, observed<45 | official, observed>=45 |
|---|---|---|---|---|
| 0.1 | 70.34 | 0.980 | 1.000 (n=4) | 0.979 |
| 0.2 | 48.11 | 0.950 | 0.750 (n=4) | 0.958 |
| 0.3 | 42.13 | 0.920 | 0.250 (n=4) | 0.948 |

Caveats: only 4 of 100 official engines have observed history < 45 — the
small-subgroup coverage numbers are extremely noisy. Overall official coverage
at alpha=0.1 is 98% on 100 engines.

## Notes

- q = 70.34 is wider than M1-8's 24.10 because it must cover the worst of
  five checkpoints (including early-life, large-RUL cutoffs) rather than one
  row per calibration point, and because n is 15, not 75.
- Empirical (official) coverage is reported separately from the formal
  guarantee; the guarantee is finite-sample over the 15-engine calibration
  cluster and the checkpoint scheme, not a universal 90% claim.