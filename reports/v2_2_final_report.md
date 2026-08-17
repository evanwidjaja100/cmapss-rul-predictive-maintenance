# V2.2 Final Report — Methodology V2.2 Scientific Repair

Date: 2026-08-17 · Repository: cmapss-rul-predictive-maintenance
Supersedes: V2.1 "CV-READY" declaration (git `0251dca`) — labeled
`SUPERSEDED BY METHODOLOGY V2.2`.

## 1. Executive summary

Methodology V2.2 repaired the two scientific leaks that invalidated V2.1's
headline numbers: calibration engines had been used as `validation_data` in
the final FD001 fit (V2.2-1), and outer-fold engines were inside training
control during CV (V2.2-2). Under the corrected nested protocol — 40/40
candidate-fold runs, a hard completeness gate, a pre-registered selection
policy, and a clean conformal recalibration on untouched engines — the V2.2
deployment model is **xgb_w90_d6** (NASA-risk champion; accuracy champion
lstm_w60_huber). Official post-hoc FD001: RMSE 26.25, NASA 60,963.8. FD004
condition-aware variant C remains the fix for the regime collapse under the
clean protocol (official post-hoc RMSE 33.66). All claims are recomputed from
saved CSVs (falsification pass) and guarded by 19 new artifact-free protocol
tests.

## 2. What was wrong (audit findings)

| ID | Severity | Finding |
|---|---|---|
| V2.2-1 | CRITICAL | V2.1 final FD001 fit passed the 15 calibration engines as `validation_data` (early stopping, checkpointing, LR schedule) |
| V2.2-2 | CRITICAL | V2.1 outer-fold engines drove early stopping AND were the fold-evaluation engines |
| V2.2-3 | CRITICAL | V2.1 CV matrix incomplete (rf_w60 fold 1 only) while claimed "8 × 5" |
| V2.2-4 | HIGH | "GRU w45 wins" claim contradicted by artifacts; selection not reproducible |
| V2.2-5 | CRITICAL | Conformal calibrated on the leaky model |
| V2.2-6/7 | HIGH/MED | Hardcoded training values; platform-dependent hashes |
| V2.2-8 | CRITICAL | FD004 variants used validation engines for early stopping + evaluation |
| V2.2-9..16 | HIGH/MED | Stale sensitivity, stale docs, no protocol tests, serving the leaky model, ordering/falsification, metadata, wording, config drift |

## 3. Repairs delivered

1. **Nested CV** (`src/rul_prediction/benchmark/v2_2.py`): per outer fold,
   68 outer-train / 17 untouched outer-eval engines; inner 58/10 splits
   (seeds 4201–4205, `random.Random(4200+fold)` on sorted IDs) control
   duration; stage-2 refits fixed-duration with NO validation data.
2. **Complete 40/40 matrix** gated by `assert_cv_complete`; summaries and
   selection blocked on partial matrices (protocol Test C).
3. **Pre-registered selection policy** (locked in `V2_2_REPAIR_PLAN.md`
   before any V2.2 CV result): PRIMARY lowest mean NASA per engine; GUARDRAIL
   pooled-SE RMSE; TIE |bias|. Applied mechanically by
   `scripts/select_v2_2_model.py` → `selection_decision.json` →
   `configs/final_model_v2_2_fd001.yaml`.
4. **Clean final fit**: 85 dev engines only, `n_estimators = median([100, 91,
   69, 97, 53]) + 1 = 92`, zero calibration contact.
5. **Clean conformal**: 15 engine scores, k = ceil((n+1)(1−α)); q(0.1) =
   66.21, q(0.2) = 44.80, q(0.3) = 41.42; formal wording limited to
   exchangeability + predefined checkpoints; arbitrary trajectories labeled
   engineering extrapolation.
6. **FD004 clean protocol**: two-stage (150/25, seed 4201) for A/B/C/D;
   variant C selected; freeze on 212 engines; 37 validation engines untouched.
7. **Configs as source of truth** (YAML-driven freeze scripts, Test E),
   **canonical hashes** (`canonical_hash.py`, Test F), **per-run metadata**
   (`fd001_outer_metadata.jsonl`), **post-hoc after calibration with
   falsification** (Test D + metric recompute).
8. **Docs live**: README/PROJECT_SPEC/CHANGELOG rewritten V2.2-first; stale
   terms removed; historical claims labeled.
9. **Serving**: `v2_predictor.py` + `app_v2.py` serve the V2.2 model with the
   recalibrated interval, per-instance fields and the extrapolation
   disclosure; no OOD language.

## 4. Final CV results (FD001, mean ± std over 5 outer folds)

| candidate | RMSE | MAE | R² | NASA total | NASA/engine |
|---|---|---|---|---|---|
| xgb_w90_d6 (**deployment**) | 28.35 ± 2.69 | 22.68 | 0.735 | 6,256.4 ± 2,726.8 | **368.02 ± 160.40** |
| lstm_w60_huber (accuracy champion) | **26.19 ± 3.44** | 21.11 | 0.767 | 7,325.8 ± 2,493.6 | 430.93 ± 146.68 |
| gru_w45_huber | 26.74 ± 3.54 | 21.26 | 0.760 | 7,496.2 ± 2,358.6 | 441.0 ± 138.7 |
| (remaining 5 candidates) | — | — | — | higher NASA or RMSE | — |

Full matrix: `experiments/v2_2/fd001_outer_fold_results.csv` + `_summary.csv`.
Note: under the clean protocol no deep candidate beats XGBoost on NASA, and
no candidate beats lstm_w60_huber on RMSE — V2.1's single-winner framing is
replaced by role-based reporting.

## 5. Final FD001 model (deployment)

- Model: XGBoost, window 90, depth 6, n_estimators 92, trained on the 85
  development engines only (`models/v2_2/fd001_xgb_w90_d6.joblib` +
  `fd001_scaler.joblib`).
- Config: `configs/final_model_v2_2_fd001.yaml` (source of truth).

## 6. Official FD001 results (post-hoc, recomputed from saved CSVs)

| metric | value |
|---|---|
| RMSE | 26.2526 |
| MAE | 21.2347 |
| R² | 0.6009 |
| NASA total | 60,963.79 |
| NASA per engine | 609.64 |
| coverage α=0.1 (q=66.21) | 99% (full-history 100%, short-history 96%) |

Predictions: `experiments/v2_2/fd001_official_predictions.csv`. These are
permanently post-hoc (labels inspected in the V2-0 audit); they are NOT
compared against the V2.1 number (23.04) as a "winner" claim — the V2.1
number was produced under leaky control.

## 7. Uncertainty calibration (final)

15 calibration engines, one max-|error| score per engine over five predefined
lifecycle checkpoints (0.25/0.45/0.65/0.80/0.95): `q(0.1) = 66.2097` (k=15),
`q(0.2) = 44.7955` (k=13), `q(0.3) = 41.4224` (k=12). Formal statement is
limited (exchangeability + predefined checkpoints); use on arbitrary
trajectories is an engineering extrapolation, disclosed in the app.

## 8. Error analysis

Post-hoc descriptive: official trajectories are truncated before failure —
`cycle.max()` is observed history, never lifetime. The frozen model
overpredicts (+19.6 cycles mean, 91% of engines), strongest on short observed
histories (< 90: +29.0…+49.4). The app's short-history risk flag is an
empirical threshold, not an OOD claim.

## 9. Sensitivity analysis

Sensor occlusion on the FINAL V2.2 model (not SHAP): most influential —
sensors 4, 11, 12, 9, 3, 20, 7. Constant sensors (1, 5, 10, 16, 18, 19)
contribute zero (consistency check). Sensor 6 (V2's earlier flag) is nearly
inert for the V2.2 model. Conclusions differ from V2 by measurement, not by
forcing.

## 10. FD004 condition-aware experiment (clean protocol)

| variant | RMSE | R² | NASA/engine | pred_std |
|---|---|---|---|---|
| A | 72.41 | −0.246 | 75,343 | 0.0 (collapse) |
| B | 72.41 | −0.246 | 75,333 | 0.0 (collapse) |
| **C (selected)** | **29.83** | **0.789** | **1,449** | 70.6 |
| D | 33.97 | 0.726 | 81,963 | 75.9 |

Selection by the same pre-declared principle (NASA per engine, then RMSE):
**C**. Unlike V2.1, D does not beat C on RMSE under the clean protocol.
Official FD004 (post-hoc): RMSE 33.6579, MAE 22.0687, R² 0.6189, NASA
1,545,798.5, pred_std 66.84. Model: `models/v2_2/fd004_gru_w45_huber_condC.keras`
(config `final_model_v2_2_fd004.yaml`).

## 11. Verification

- Full test suite: **134 passed** (incl. 19 V2.2 protocol tests A–H + duration
  rules; 4 rewritten serving tests; legacy golden test passes).
- Artifact-free CI subset: **130 passed** (4 `needs_artifacts` deselected) —
  verified locally, same command as CI.
- Falsification: selection re-derived from fold CSVs (Test D); official
  metrics recomputed from saved prediction CSVs; config drift asserted; CV
  numbers re-verified.
- App import smoke: `python -c "import app_v2"` passes.
- Self-audit rounds 1 & 2 (artifacts + live-doc grep): PASS (see
  `V2_2_REPAIR_PLAN.md`).
- Exit checklist (20 criteria): all DONE.