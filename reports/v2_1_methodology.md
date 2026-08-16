# Methodology V2.1 — Repair and Results

Date: 2026-08-15 · Plan: `V2_1_REPAIR_PLAN.md` · Base: Methodology V2 (93 tests)

## Why V2.1 exists

The V2 methodology had four critical scientific issues found by audit:

1. **Test trajectories are truncated before failure.** `cycle.max()` of an
   official test engine was called its "lifetime" (e.g. "lifetime < 128",
   "short-lifetime engines", "OOD"). The correct quantity is
   `implied_failure_cycle = observed_cycles + true_rul`.
2. **Selection overfit a single 15-engine validation set** (~45 runs) and
   conformal calibration treated 75 correlated rows as n = 75.
3. **FD004's operating regimes were never modeled** — the collapse to a
   constant was reported but not investigated.
4. **Serving presented observed-history length as lifetime/OOD.**

## What changed

| area | V2 | V2.1 |
|---|---|---|
| test trajectory | "lifetime", OOD < 128 | observed history + `implied_failure_cycle`; `history_is_padded` (objective), empirical short-history risk flag (never OOD) |
| model selection | 1×15-engine validation set, 45 runs | 5-fold engine-group CV (seed 42), 8 bounded candidates, per-fold scaler |
| lifecycle cutoffs | 0.50/0.65/0.80/0.90/0.95 | 0.25/0.45/0.65/0.80/0.95 (balanced, fixed pre-comparison) |
| conformal | 75 correlated rows, q=24.10 | engine cluster n=15, max-|err| per engine, k=ceil((n+1)(1-a)) clamped, q=70.34 |
| FD004 | global scaler → collapse (unfixed) | KMeans(k=6) per-regime scalers → variance restored |
| final config | hardcoded in scripts | YAML configs (hashes, CV metrics, software) |
| official test | "exactly once" (false), then post-hoc | post-hoc, always labeled post-hoc |

## Results

### FD001 (frozen gru_w45_huber, 85 dev engines, post-hoc official)

- CV (5-fold): RMSE 24.45±3.62, R2 0.805, NASA 5,828±4,451, bias −0.25.
- Official: RMSE 23.04, MAE 15.61, R2 0.693, NASA 6,700.59 — 11.5x NASA
  improvement over V2 (77,387.53).
- Engine-cluster interval alpha=0.1: q=70.34; official coverage 98%.

### FD001 error analysis (corrected semantics)

- `implied_lifetime_lt_128` is **empty** on the official test: the V2 claim
  "44 engines with lifetime < 128 carry 99.8% of NASA" has no lifetime-based
  counterpart — it was an observed-history artifact.
- Overprediction concentrates in short observed history: observed 45–127
  (40 engines) mean err +17.4 / 87.5% of NASA; observed ≥ 128 (56 engines)
  mean err −4.0 / 1.7% of NASA. Padded (observed < 45, n=4) mean err +45.9.

### FD004 (variant C, per-regime scalers)

- A/B (global scaler ± settings) collapse: pred std 0.0, RMSE ~72, R2 −0.23.
- C restores variance: RMSE 29.37, R2 0.795, NASA 46,165 (55x vs A).
- D (C + one-hot regime): RMSE 27.22, R2 0.824 but NASA 68,981 — not selected.
- Frozen C official: RMSE 33.83, R2 0.615, NASA 1,345,518 (1.9x better than
  the V2-11 baseline collapse).

## Reproducibility

- All split/manifest/fold artifacts committed (`experiments/splits/`,
  `experiments/v2_1/`); results CSVs committed; frozen weights gitignored
  (`models/`) but reproducible via the committed scripts + configs.
- Configs: `configs/final_model_v2_1_fd001.yaml`, `final_model_v2_1_fd004.yaml`.
- Scripts: `build_v2_1_manifests.py`, `run_v2_1_cv.py`, `run_v2_1_freeze.py`,
  `calibrate_v2_1_conformal.py`, `analyze_v2_1_errors.py`,
  `run_v2_1_fd004.py`, `run_v2_1_fd004_freeze.py`.
- Tests: `tests/test_v2_1_methodology.py` (16), `test_v2_1_fd004_condition.py`
  (5), `test_v2_serving.py` (4).

## Superseded V2 conclusions

Every V2 conclusion is preserved and labeled `SUPERSEDED BY METHODOLOGY V2.1`
in the V2 reports (see `reports/v2_*.md` banners). V2 remains historical
record; V2.1 is the operative methodology.