# Methodology M2 — Repair and Results

Date: 2026-08-15 · Plan: `M2_REPAIR_PLAN.md` · Base: Methodology M1 (93 tests)

## Why M2 exists

The M1 methodology had four critical scientific issues found by audit:

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

| area | M1 | M2 |
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
  improvement over M1 (77,387.53).
- Engine-cluster interval alpha=0.1: q=70.34; official coverage 98%.

### FD001 error analysis (corrected semantics)

- `implied_lifetime_lt_128` is **empty** on the official test: the M1 claim
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
  the M1-11 baseline collapse).

## Reproducibility

- All split/manifest/fold artifacts committed (`experiments/splits/`,
  `experiments/m2/`); results CSVs committed; frozen weights gitignored
  (`models/`) but reproducible via the committed scripts + configs.
- Configs: `configs/final_model_m2_fd001.yaml`, `final_model_m2_fd004.yaml`.
- Scripts: `build_m2_manifests.py`, `run_m2_cv.py`, `run_m2_freeze.py`,
  `calibrate_m2_conformal.py`, `analyze_m2_errors.py`,
  `run_m2_fd004.py`, `run_m2_fd004_freeze.py`.
- Tests: `tests/test_m2_methodology.py` (16), `test_m2_fd004_condition.py`
  (5), `test_m1_serving.py` (4).

## Superseded M1 conclusions

Every M1 conclusion is preserved and labeled `SUPERSEDED BY METHODOLOGY M2`
in the M1 reports (see `reports/m1_*.md` banners). M1 remains historical
record; M2 is the operative methodology.

## Exit checklist (final, 2026-08-15)

| # | check | result |
|---|---|---|
| 1 | Full test suite | 115 passed |
| 2 | CI subset (`-m "not needs_artifacts"`) | 111 passed, 4 deselected |
| 3 | App smoke (`python -c "import app_m1"`) | OK |
| 4 | Falsification: official metrics recomputed from prediction CSVs (FD001 + FD004) | exact match |
| 5 | Stale-claim grep (OOD/lifetime<128/99.8%/47.7%/85.7%/exactly-once/sealed/SHAP mislabel) | live files clean; historical files bannered |
| 6 | M2_REPAIR_PLAN.md statuses | R1–R20 DONE |
| 7 | Configs read by scripts, hashes verified | FD001 config integrity checks pass in freeze script |
| 8 | Git tree | clean (commits `67a0e58`, `3d12bc7`; only session dir `.opencode/` untracked) |
| 9 | `requires-python` truthful | `>=3.11,<3.13` |
| 10 | Upstream attribution | README + `THIRD_PARTY_NOTICES.md` |

**CV-READY.** Methodology M2's model selection (FD001), uncertainty
calibration, corrected error analysis, and FD004 condition-aware modeling are
complete, reproducible from committed code/configs/artifacts, and verified by
the exit checklist above. Official-test numbers are labeled post-hoc
throughout. M2 supersedes Methodology M1, which remains as historical
record.