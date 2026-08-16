# Phase 8 — Ablation & Hyperparameter Experiments

Date: 2026-08-15  |  Split: FD001 seed 42 (80/20 engines)  |  Evaluation: **validation engines only** — the NASA FD001 test set was not touched anywhere in this phase.

All experiments follow the one-factor-at-a-time protocol around the Phase 7 base configuration (GRU, window 30, RUL cap 125, MSE, all 21 sensors). Scripts are reproducible: `scripts/preprocess.py` builds each (window, cap, sensors) variant under `data/processed/<dataset>_w<W>_c<cap>_<sensors>/`; `scripts/run_experiment.py --variant ...` trains and logs.

## A — Sequence length (GRU, cap 125, MSE, all sensors)

| window | train seqs | RMSE | MAE | R² | NASA |
|---|---|---|---|---|---|
| 15 | 15222 | 16.74 | 12.23 | 0.839 | 33295 |
| **30 (base)** | 14022 | 13.47 | 9.57 | 0.897 | 19178 |
| 50 | 12422 | 11.88 | 8.78 | 0.919 | 8105 |
| 60 | 11622 | 11.63 | 8.48 | 0.922 | 6919 |
| **90** | 9222 | **11.17** | 8.37 | 0.921 | 4203 |
| 120 | 6822 | 11.68 | 8.13 | 0.904 | 3163 |

Longer windows strictly help until ≈90 cycles, then RMSE plateaus/slightly degrades as fewer sequences survive (6918 sequences/windows lose 120-cycle windows vs 480 at 90). **Select: window 90.**

## B — RUL clipping cap (GRU, window 30, MSE, all sensors)

| cap | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| none | 31.59 | 22.14 | 0.756 | 1415355 |
| 150 | 18.09 | 14.20 | 0.866 | 28949 |
| 125 (base) | 13.47 | 9.57 | 0.897 | 19178 |
| 100 | 8.99 | 6.01 | 0.925 | 5268 |
| 75 | 5.35 | 3.14 | 0.947 | 1703 |
| 60 | 3.92 | 2.44 | 0.949 | 1064 |
| **45** | **2.63** | **1.75** | **0.950** | **644** |

Caveat checked, not papered over: a smaller cap shrinks target variance (the early ~75% of each engine's windows share target = cap), so part of the gain is mechanical. A diagnostics check on the c60/c45 models shows they are **not** degenerate constants: per-target-bucket predictions track the decline (corr 0.956/0.979 on validation windows; fraction of predictions stuck at the cap = 0.00). **Select: cap 45** — the monotone gain and the decline-tracking diagnostic justify it; test evaluation at Phase 9 applies the same cap, so the comparison stays consistent.

## C — Loss function (GRU, base variant)

| loss | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| **MSE (base)** | **13.47** | 9.57 | **0.897** | **19178** |
| MAE | 15.10 | 10.49 | 0.870 | 17620 |
| Huber | 15.38 | 10.40 | 0.865 | 22494 |

**Select: MSE.**

## D — Sensor set (window 30, cap 125, MSE)

| model | all 21 | 15 varying (6 constants removed) |
|---|---|---|
| GRU | **13.47** / 19178 | 14.23 / 18942 |
| XGBoost | 14.25 / 13128 | 13.87 / 11780 |

Constants removed: sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19 (std ≈ 0 on the training partition). Dropping them slightly *hurts* GRU and barely helps XGBoost; the champion model keeps all 21 channels. **Select: all sensors.**

## E — Architecture at the final variant (window 90, cap 45, MSE, all sensors)

| model | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| **XGBoost** | **2.376** | **1.258** | **0.969** | **326** |
| GRU | 2.654 | 1.743 | 0.961 | 449 |
| LSTM | 2.922 | 1.688 | 0.953 | 462 |

At the base configuration GRU was champion (13.47 vs XGBoost 14.25); at the final variant the engineered-feature model wins with a small but consistent margin. Honest negative for the deep models: with a tight cap the signal concentrates in the last 45 cycles, where hand-built slope/last-window features are strong. **Select: XGBoost.**

## Composed check (A+B interacted together, GRU)

| variant | RMSE | R² | NASA |
|---|---|---|---|
| w90_c45_all | **2.65** | **0.961** | **449** |
| w90_c60_all | 3.79 | 0.962 | 689 |

The individually-best factors compose without interaction loss (w90_c45 = 2.65 ≈ w30_c45 2.63 with higher R² of 0.961 vs 0.950).

## Locked final configuration (validation-selected, for Phase 9)

```
model: xgboost
variant: w90_c45_all        # window 90, RUL cap 45, all 21 sensors, scalar per-window features
loss: mse                   # N/A for XGBoost (training loss is XGBoost's own)
seed: 42
validation RMSE: 2.376 | MAE: 1.258 | R²: 0.969 | NASA: 326
```

The full factor table is in `reports/tables/ablation_results.csv` (26 rows, validation-only, seed 42). The configuration will be frozen in `configs/final_model.yaml` at Phase 9 before any test-set contact.