> **SUPERSEDED BY METHODOLOGY V2.1** — this report documents the V2
> methodology as historical record. V2.1 corrects the lifetime semantics
> (observed history vs implied failure cycle), replaces the single 15-engine
> validation with 5-fold engine-group CV, recalibrates conformal at the
> engine level, and fixes the FD004 collapse with per-regime scaling.
> See V2_1_REPAIR_PLAN.md and reports/v2_1_methodology.md.

# Phase V2-5 — Model Freeze & Post-Hoc Official FD001 Evaluation

Date: 2026-08-15  |  Split: FD001 seed 42 (70/15/15 engines)  |  Target: **raw RUL (no cap)**

## Frozen model

Configuration = the V2-4-selected run `v2_gru_w45_losshuber_s42`, exactly: **GRU, window 45, huber loss, batch 256, early stopping patience 8, seed 42**, raw-RUL target, train-only scaler (`data/processed/v2/FD001/raw/FD001_scaler.joblib`). The model was retrained with the identical procedure (train engines only; early stopping on validation-engine sequences; calibration engines untouched) and saved to `models/v2_frozen_gru_w45_huber.keras` (gitignored).

Retraining reproduced the selection metrics **bit-exact**: validation RMSE 13.7406, MAE 9.6908, R² 0.8768, NASA 200.01 — the freeze is a deterministic reproduction of the selected run, not a different model wearing its name.

## Post-hoc official FD001 test evaluation

**IMPORTANT — post-hoc status:** the official RUL labels were inspected during the V2-0 audit. This evaluation is therefore labeled **post-hoc**, never "exactly once". It is the first V2 contact with the official test set.

| metric | value |
|---|---|
| engines | 100 (1–100, ids verified) |
| RMSE | **29.0377** |
| MAE | **19.1715** |
| R² | **0.5117** |
| NASA total / mean | **77387.53 / 773.88** |
| |err| < 13 cycles (good prediction) | 54.0% |
| padded engines (lifetime < 45 cycles) | 4 (handled by the shared padding/mask builder) |

Rows: `reports/tables/v2_fd001_official.csv` (metrics), `experiments/v2_fd001_official_predictions.csv` (per-engine; gitignored). Metrics recomputed independently from the predictions file — identical.

### Baselines on the official test set (raw RUL)

| model | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| **frozen GRU w45 huber** | **29.04** | **19.17** | **0.512** | 77387 |
| constant = train raw-RUL mean (106.7) | 51.95 | 40.43 | −0.563 | 241062 |
| constant = mean of model predictions | 42.50 | 35.91 | −0.046 | 26681 |
| legacy raw-RUL model (V2-0 audit, `FD001_final_test_results.json`) | 50.37 | 39.05 | −0.469 | 19261.8 |

The frozen model beats every baseline on RMSE/MAE/R². Its NASA score is worse than the legacy raw model's despite far better central accuracy — see below.

### Honest interpretation of the validation → official gap (13.74 → 29.04)

1. **Official test distribution differs from the validation manifest.** Official test RUL values span 7–145 (mean 75.5, median 86) — the entire official test set sits in the lower half of the training RUL range (train raw RUL up to 361). The validation manifest rows (cutoff fractions 0.50–0.95) include larger raw-RUL cases the official test never contains, and vice versa.
2. **Late bias.** Mean error is +8.9 cycles (late). 11 engines are missed late by >50 cycles (worst +110.6), none are missed early by >50. The model is systematically optimistic on the official test.
3. **NASA is dominated by tail errors, not central accuracy.** With raw targets, a single +110-cycle late miss scores ≈ exp(11.06)−1 ≈ 63,000 points. The five worst late misses contribute 74,749 of the 77,388 total (96.6%). The constant baselines "win" on NASA only because their predictions cannot be far late on high-RUL engines.
4. **R² scale.** R² 0.512 is computed against official-test target variance; the validation R² (0.877) was computed against a different sample. Neither number is wrong; they answer different questions.

This is a genuine finding, not a bug: raw-RUL NASA is an asymmetric exponential score, and a huber/MSE-trained model has no incentive to avoid rare large late errors. Conformal calibration (V2-8) and asymmetric/conservative adjustment are the natural mitigations; V2-6 (error analysis) will characterize the late-miss engines before V2-8.

## Artifacts

- `models/v2_frozen_gru_w45_huber.keras` — frozen model (gitignored).
- `reports/tables/v2_fd001_official.csv` — tracked metrics row.
- `experiments/v2_fd001_official_predictions.csv` — per-engine predictions (gitignored).
- `scripts/run_v2_freeze.py` — reproducible freeze + post-hoc evaluation (retrains from scratch; asserts ≤1.5 RMSE drift vs the selected run).

Next: **Phase V2-6 — error analysis** of the validation and official-test prediction residuals (per-engine profiles, late-bias characterization).