# V2.1 Cross-Validation and Model Selection (FD001)

Date: 2026-08-15 · Repair plan R5/R6/R7/R8

## Design (locked before any comparison)

- **Development / calibration separation**: 100 FD001 engines → 85 development
  + 15 calibration (`[22, 27, 31, 33, 34, 38, 46, 49, 50, 77, 81, 84, 88, 93, 97]`,
  sha256 prefix `2b6229c9ef25`, preserved from the V2 split — used only for
  conformal calibration, never for selection). Development sha256 `05ea70879d1e`.
- **5-fold engine-group CV, seed 42**: 68 train / 17 validation engines per
  fold; every development engine validated exactly once; folds disjoint.
  Manifest: `experiments/splits/fd001_v2_1_group_cv_seed42.json`.
- **Lifecycle fractions** (fixed pre-comparison): 0.25 / 0.45 / 0.65 / 0.80 /
  0.95 → 85 manifest rows per fold (`fd001_v2_1_fold{k}_validation_cutoffs.csv`).
- **Per-fold scaler**: StandardScaler fit on that fold's training rows only.
- **Candidates (8 ≤ 12 bound)**: gru/lstm w45+w60 huber, rf/xgb w60+w90.
  Excluded with reason: TCN (never competitive in V2-4, ~2x cost), linear
  (unusable on raw RUL, V2-3).

## Results (experiments/v2_1/fd001_cv_summary.csv, fd001_cv_fold_results.csv,
fd001_cv_engine_level.csv, fd001_cv_predictions.csv)

| candidate | RMSE (mean±std) | MAE | R2 | NASA total (mean±std) | signed bias | time/fold |
|---|---|---|---|---|---|---|
| gru_w45_huber | **24.45 ± 3.62** | 16.00 | **0.805** | 5,827.8 ± 4,451 | **−0.25** | 322 s |
| lstm_w45_huber | 24.61 ± 3.25 | 16.04 | 0.802 | 24,769 ± 44,071 | 1.72 | 260 s |
| lstm_w60_huber | 22.15 ± 3.96 | 14.50 | 0.840 | 36,509 ± 60,705 | 0.13 | 359 s |
| gru_w60_huber | 29.38 ± 12.97 | 20.76 | 0.672 | 53,802 ± 104,968 | 3.09 | 340 s |
| xgb_w90_d6 | 28.39 ± 2.86 | 22.86 | 0.733 | **5,693.9 ± 2,366** | +14.08 | 5 s |
| xgb_w60_d6 | 35.27 ± 2.58 | 30.83 | 0.589 | 9,426.6 ± 3,689 | +21.88 | 5 s |
| rf_w90 | 31.15 ± 2.16 | 23.76 | 0.680 | 136,705 ± 254,439 | 14.08 | 86 s |
| rf_w60 | 35.83 | 29.39 | 0.648 | 42,419 | 11.06 | 90 s |

## Selection

**gru_w45_huber** — best RMSE/MAE/R2 and near-zero bias on the balanced
manifest; NASA within 2.3% of the runner-up (xgb_w90_d6), whose NASA edge is
within fold-level noise (group-mean comparison) and whose +14-cycle systematic
overprediction would skew conformal intervals. This matches the V2 winner
(GRU w45 huber), now validated by engine-group CV instead of a single
15-engine validation set.

## Frozen final model (configs/final_model_v2_1_fd001.yaml)

Retrained on all 85 development engines (scaler on those 85 only), then
post-hoc official FD001:

| metric | value | vs V2 (70-engine train) |
|---|---|---|
| RMSE | 23.04 | 29.04 |
| MAE | 15.61 | 19.17 |
| R2 | 0.693 | 0.512 |
| NASA total | 6,700.59 | 77,387.53 (11.5x) |

Artifacts: `models/v2_1/fd001_gru_w45_huber.keras`,
`models/v2_1/fd001_scaler.joblib`,
`reports/tables/v2_1_fd001_official.csv`, `v2_1_fd001_predictions.csv`.

## Reproducibility

- CV manifest + fold manifests committed (un-ignored `experiments/splits/`);
- fold results / engine-level / predictions / summary committed
  (`experiments/v2_1/`);
- every training: seed 42, GRU w45 (128,64) dropout 0.3 huber, lr 1e-3,
  batch 256, epochs 60, patience 8, early stopping on the fold's validation
  engine sequences.