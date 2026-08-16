> **SUPERSEDED BY METHODOLOGY V2.1** — this report documents the V2
> methodology as historical record. V2.1 corrects the lifetime semantics
> (observed history vs implied failure cycle), replaces the single 15-engine
> validation with 5-fold engine-group CV, recalibrates conformal at the
> engine level, and fixes the FD004 collapse with per-regime scaling.
> See V2_1_REPAIR_PLAN.md and reports/v2_1_methodology.md.

# Phase V2-4 — Raw-RUL Window & Hyperparameter Ablation

Date: 2026-08-15  |  Split: FD001 seed 42 (70/15/15 engines)  |  Evaluation: **75 fixed validation-manifest rows only** — the NASA FD001 test set was not contacted anywhere in this phase. Calibration engines (75 rows) were not used.

Target = **raw RUL (no cap)** throughout. One-factor-at-a-time around the V2-3 base configuration (window 30, MSE). Every run uses the shared runner (`src/rul_prediction/benchmark/v2.py`) and the same 75 validation rows, so RMSE/MAE/R²/NASA totals are directly comparable. `scripts/run_v2_ablation.py` reproduces the whole matrix (idempotent; results in `reports/tables/v2_ablation_results.csv`). CPU-bound note: deep models were run at all windows (15–90); the original plan capped deep windows at 45 for time, but the w45 trend made the full sweep worth ~2 h, so the sweep was completed.

## A — Window ablation, classical models (raw RUL)

| window | mean RMSE | linear RMSE | rf RMSE | xgboost RMSE | rf R² | xgboost R² | rf NASA | xgboost NASA |
|---|---|---|---|---|---|---|---|---|
| 15 | 66.45 | 694.08 | 43.51 | 36.35 | −0.235 | 0.138 | 17636 | 19974 |
| 30 (base) | 66.45 | 465.95 | 30.33 | 33.45 | 0.400 | 0.270 | 5705 | 7008 |
| 45 | 66.45 | 295.45 | 38.33 | 41.14 | 0.042 | −0.105 | 7202 | 8485 |
| 60 | 66.45 | 187.16 | 32.21 | 34.06 | 0.323 | 0.243 | 4626 | 3809 |
| 90 | 66.45 | 90.44 | **18.55** | **19.81** | **0.776** | **0.744** | **456** | **532** |

The mean baseline is a constant (train raw-RUL mean ≈ 106.7) and never varies. Linear regression is unusable on raw RUL (NASA totals explode to 10⁴⁵ at short windows); its error shrinks monotonically with window length but never becomes competitive. Tree ensembles follow the same long-window trend: **w90 is the classical optimum** (rf 18.55, xgboost 19.81 RMSE), roughly a 40% RMSE gain over w30.

## B — Window ablation, deep models (raw RUL)

| window | lstm RMSE | gru RMSE | tcn RMSE | lstm R² | gru R² | lstm NASA | gru NASA |
|---|---|---|---|---|---|---|---|
| 15 | 28.64 | 29.53 | 31.72 | 0.465 | 0.431 | 20696 | 25282 |
| 30 (base) | 24.35 | 24.66 | 32.52 | 0.613 | 0.603 | 3738 | 24538 |
| 45 | 15.75 | 14.85 | 32.41 | 0.838 | 0.856 | 569 | 397 |
| 60 | **13.98** | **13.89** | 27.15 | **0.873** | **0.874** | **240** | 269 |
| 90 | 17.01 | 17.35 | 28.74 | 0.811 | 0.804 | 520 | 513 |

Recurrent models improve steeply with window length through w60, then degrade at w90 (more masked/padded history hurts the recurrent path; classical ensembles, which tolerate it, keep improving to w90). TCN never reaches the recurrent architectures on this problem. GRU and LSTM are effectively tied at w60; GRU trains ~2× faster (325 s vs 633 s).

## C — XGBoost hyperparameters (w30)

| config | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| depth 3 | 41.93 | 36.05 | −0.147 | 27807 |
| depth 6 (base) | 33.45 | 28.12 | 0.270 | 7008 |
| depth 9 | **30.78** | **25.30** | **0.382** | **4558** |
| n_estimators 300 | 33.45 | 28.12 | 0.270 | 7008 |

Deeper trees help at w30, but the w90 sweep (B/A) already beats any w30 depth. n_estimators 300 and 500 give identical metrics (early stopping converged before 300 either way), confirming 500 is not overkill. Depth 9 at w90 (21.40 RMSE) does not beat depth 6 at w90 (19.81) — depth 6, w90 stands.

## D — LSTM hyperparameters (w30)

| config | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| dropout 0.2 | 49.63 | 35.33 | −0.608 | 492228 |
| dropout 0.3 (base) | 24.35 | 17.38 | 0.613 | 3738 |
| dropout 0.4 | 48.36 | 33.44 | −0.526 | 362267 |
| **loss huber** | **21.31** | **14.07** | **0.704** | **4384** |

Dropout is sharply non-monotone — the base 0.3 is a strong local optimum; 0.2 and 0.4 both collapse. Huber loss beats MSE (21.31 vs 24.35 RMSE), which motivates the loss check at the leading windows below.

## E — Huber loss at the leading windows (lstm/gru)

| config | RMSE | MAE | R² | NASA | train time |
|---|---|---|---|---|---|
| lstm w45 + huber | 15.15 | 11.04 | 0.850 | 309 | 341 s |
| gru w45 + huber | **13.74** | 9.69 | **0.877** | **200** | 339 s |
| gru w60 + huber | 14.20 | **9.37** | 0.868 | 747 | 762 s |

Huber at w45 confirms the trend (gru 14.85 → 13.74 RMSE, NASA 397 → 200). At w60, huber's RMSE stays close (14.20) and MAE is best overall (9.37), but the NASA score degrades sharply (747 vs 200 at w45) — a few large misses dominate the NASA score, and w45+huber avoids them.

## Selection

**Final architecture: GRU, window 45, huber loss, batch 256, early stopping patience 8** — `v2_gru_w45_losshuber_s42`.

| finalist | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| **gru w45 huber** | **13.74** | 9.69 | **0.877** | **200.01** |
| gru w60 mse | 13.89 | 10.97 | 0.874 | 269.34 |
| lstm w60 mse | 13.98 | 9.98 | 0.873 | 240.21 |
| gru w60 huber | 14.20 | 9.37 | 0.868 | 747.20 |
| rf w90 | 18.55 | 16.71 | 0.776 | 455.70 |
| xgboost w90 | 19.81 | 17.30 | 0.744 | 531.59 |

Chosen on the primary criterion (validation RMSE) with NASA as secondary: it has the lowest RMSE (13.74) and the lowest NASA total (200.01, mean error score 2.67 per engine) of all 45 runs, at 5.7 min training time. The w60 GRU variants trade RMSE for marginal MAE gains at the price of NASA stability, so w45+huber wins. Deep (13.74) beats the best classical (rf w90, 18.55) by ~26% RMSE and 2.3× better NASA.

Run the freeze at V2-5 with exactly these parameters: `gru`, window 45, `loss=huber`; target remains raw RUL; same 75-row validation manifest; post-hoc evaluation on the official FD001 test set (labeled post-hoc, not "exactly once").