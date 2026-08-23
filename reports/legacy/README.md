# Legacy Maintenance-Horizon (Cap-45) Experiment

This directory labels the historical **45-cycle maintenance-horizon RUL
experiment** produced by the original Phase 1–10 implementation.

## What is legacy here

| Artifact | Role |
|---|---|
| `configs/final_model.yaml` / `configs/legacy_cap45_model.yaml` | Frozen cap-45 configuration (window 90, XGBoost, `max_rul: 45`) |
| `reports/phase8_ablation.md` | Cap sweep that selected 45 (target-range-changing ablation; not repeated in M1) |
| `reports/phase9_final_evaluation.md` | Official FD001 test evaluation against clipped truth |
| `reports/phase10_serving.md` | HTTP/batch serving + golden-file verification (service retained in M1) |
| `reports/tables/ablation_results.csv` | All Phase 8 validation metrics (overlapping-window protocol) |
| `experiments/results.csv` | Raw Phase 5–8 experiment log (overlapping-window protocol) |
| `experiments/FD001_final_test_results.json` | Clipped (RMSE 2.402) **and** raw-RUL (RMSE 50.37) official metrics |
| `experiments/splits/FD001_seed42.json` | 80/20 engine split (legacy) |
| `data/processed/FD001_w90_c45_all/` | Cap-45 processed artifacts incl. `FD001_test_predictions.csv` |
| `models/final/FD001_final_model.joblib` | Frozen cap-45 model |

## Why it is labeled legacy

- Primary target is `min(raw_rul, 45)` — a truncated target, not raw NASA RUL.
- Validation used thousands of overlapping windows per engine and NASA totals
  summed over different sample counts per window setting — not comparable to
  the official one-terminal-prediction-per-engine test task.
- The cap itself was selected by minimizing RMSE, which changes the target
  definition between experiments.

## Status in Methodology M1

- M1 primary target: **raw RUL** (`raw_rul = max_cycle - current_cycle`, no cap).
- The cap-45 task is retained only as a clearly separated **secondary
  maintenance-horizon experiment**. Its metrics are never mixed into M1
  benchmark tables.
- The HTTP/batch serving service built around the legacy model is retained;
  it is a deployment layer, not a benchmark.