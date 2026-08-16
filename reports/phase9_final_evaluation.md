# Phase 9 — Final Model Freeze & Official Test-Set Evaluation

Date: 2026-08-15

## Audit trail (order of operations)

1. `configs/final_model.yaml` frozen **before any test-set contact** — contains the Phase 8 validation-selected configuration and its pre-test validation metrics (RMSE 2.376).
2. `scripts/final_evaluation.py` committed together with the config (commit `02ccbbd`).
3. Harness run: reproducibility pass on **validation engines** reproduced the frozen metrics **exactly** (RMSE 2.3760247 vs frozen 2.376, tolerance 1e-3) → gate passed → the official test set was contacted at this commit, by this script, for the first time (historically labeled "exactly once" — see correction below).
   > **Correction (V2-12):** "contacted exactly once" no longer holds — the official RUL labels were re-inspected during the V2 audits (V2-0 inventory, V2-5 post-hoc evaluation) and the frozen predictions were reused by serving verification. All FD001 official results are therefore reported as **post-hoc official benchmark** from Methodology V2 onward; see `AUDIT_V2.md` Issues 1 and 7.
4. Results written to `experiments/FD001_final_test_results.json` + `data/processed/FD001_w90_c45_all/FD001_test_predictions.csv`.

No test-derived signal fed back into any model, preprocessing choice, or config (all ablations in Phase 8 were validation-only; the config file was committed before the run).

## Frozen configuration (configs/final_model.yaml)

```
model: xgboost        variant: w90_c45_all      window: 90
max_rul: 45           sensors: all              seed: 42
xgboost: n_estimators=500, max_depth=6, lr=0.05, subsample=0.8,
         colsample_bytree=0.8, early_stopping_rounds=30 (eval_set = validation engines only)
```

## Official test-set results — FD001 (100 units, evaluated once)

**Metrics on RUL clipped at the frozen cap (45)** — the convention used throughout the ablations, so the comparison is consistent:

| metric | value |
|---|---|
| RMSE | **2.402** |
| MAE | **1.455** |
| R² | **0.962** |
| NASA score | **14.77** |

**Metrics on raw (unclipped) test RUL** — reported for full transparency only; predictions are capped at 45 by design (frozen config), so raw-RUL RMSE (50.37) and R² (−0.47) look bad *by construction*: the model never predicts above the cap while raw RUL reaches 145. The clipped convention is the primary result.

## Details & honesty notes

- **Short units:** 26 of 100 test units have lifetime < 90 cycles (min 31). Their single available trajectory is left-padded with zeros (= scaled mean) so the window ends at the last observed cycle; flagged in the predictions CSV (`padded_short`). Verified by unit tests (`tests/test_final_evaluation.py`).
- Prediction sanity: mean 37.4 (capped), min 6.57 vs true min 7, 0 predictions above 45.
- The NASA clipped score (14.8) reflects that late predictions (predicting more RUL than true) are essentially absent — the cap prevents over-prediction beyond 45 and the model tracks the final-cycle decline tightly.
- Artifacts: `models/final/FD001_final_model.joblib` (regenerable from `scripts/final_evaluation.py`), test predictions CSV, results JSON.

## Repository state
- Commit `02ccbbd` froze config + harness pre-test; this phase's report/changelog commit follows.
- 47 pytest tests pass.