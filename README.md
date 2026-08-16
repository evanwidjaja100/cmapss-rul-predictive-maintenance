# cmapss-rul-predictive-maintenance

**Leakage-Safe Remaining Useful Life Prediction for Predictive Maintenance Using NASA C-MAPSS (FD001)**

A professional, reproducible RUL prediction system for turbofan-engine prognostics, built phase-by-phase with strict leakage safety: engine-level validation splits, training-engine-only preprocessing fitting, and an official test set that is contacted **exactly once**, after the final configuration is frozen.

**Final official test-set result (FD001): RMSE 2.40 | MAE 1.46 | R² 0.962 | NASA score 14.8** (RUL clipped at the frozen cap 45), served by a dependency-light HTTP inference service.

---

## The story in five lines

1. 100 turbofan engines with 21 sensor channels → sliding windows → engineered features → ML models.
2. Every modeling decision (features, windows, clipping, losses, architectures) was chosen on an **engine-disjoint validation split only**; Phase 8 ran 26 controlled ablation experiments (full table in `reports/tables/ablation_results.csv`).
3. The winner — **XGBoost on 90-cycle windows, RUL clipped at 45, all 21 sensors, seed 42** — was frozen in `configs/final_model.yaml` *before* any test contact.
4. The official test set was evaluated exactly once: **RMSE 2.402, NASA 14.77** (clipped convention).
5. The frozen model is served as an HTTP API / batch CLI whose output is golden-file-tested to be identical to the one-time evaluation.

## Pipeline & architecture

```
data/raw (NASA txt)                    ── downloaded once, never modified
   │  scripts/preprocess.py --validate-only
   ▼
validation: shape/values/engine counts (pytest-guarded)
   │  python -m rul_prediction.data.splitting --dataset FD001     (80/20 engines, seed 42, pinned JSON)
   ▼
engine-disjoint train/validation split (no window ever crosses an engine)
   │  scripts/preprocess.py --dataset FD001 [--window W --max-rul C --sensors all|varying]
   ▼
data/processed/FD001_<variant>/  scaled windows (float32) + train-only scaler + metadata
   │  scripts/run_experiment.py --model {mean,linear,rf,xgboost,lstm,gru,tcn} --variant ...
   ▼
experiments/results.csv   (validation-only metrics; ablations table derived)
   │  scripts/final_evaluation.py --config configs/final_model.yaml
   ▼
frozen config → reproducibility gate → ONE-TIME official test evaluation
   │  scripts/serve.py serve | batch
   ▼
HTTP /predict + /health  (stdlib, no new deps)  ── golden-tested vs Phase 9 outputs
```

## Scientific guardrails (the non-negotiables)

1. Official test data never influences model configuration.
2. Test data never used as `validation_data`.
3. Scalers fitted on training engines only.
4. Windows from one engine never split across train/validation.
5. Hyperparameters chosen on validation engines only.
6. Test evaluation happens only after configuration is frozen (committed config + reproducibility gate that aborts on drift).
7. Reproducible seeds; metrics always computed programmatically.
8. No fabricated results; methodology changes documented in CHANGELOG.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full specification and [CHANGELOG.md](CHANGELOG.md) for the per-phase record.

## Reproduce everything (Windows)

```bash
# 0. Environment (Python >= 3.11, tested on 3.12; TensorFlow supports 3.9-3.12)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

# 1. Validate raw data (official NASA FD001 files in data/raw)
.venv\Scripts\python.exe scripts\preprocess.py --dataset FD001 --validate-only

# 2. Engine-level split (idempotent, pinned to seed 42)
.venv\Scripts\python.exe -m rul_prediction.data.splitting --dataset FD001

# 3. Build processed artifacts (base variant; flags reproduce every Phase-8 variant)
.venv\Scripts\python.exe scripts\preprocess.py --dataset FD001
.venv\Scripts\python.exe scripts\preprocess.py --dataset FD001 --window 90 --max-rul 45 --sensors all

# 4. Experiments & ablations (validation-only; see scripts/run_experiment.py --help)
.venv\Scripts\python.exe scripts\run_experiment.py --dataset FD001 --model xgboost --variant w90_c45_all
.venv\Scripts\python.exe scripts\build_ablation_table.py      # -> reports/tables/ablation_results.csv

# 5. Final: reproducibility gate + one-time official test evaluation
.venv\Scripts\python.exe scripts\final_evaluation.py --config configs\final_model.yaml

# 6. Serve (HTTP on :8000, or batch mode with optional RUL metrics)
.venv\Scripts\python.exe scripts\serve.py serve --port 8000
.venv\Scripts\python.exe scripts\serve.py batch --input data\raw\test_FD001.txt --rul data\raw\RUL_FD001.txt
```

Every gitignored artifact (`data/processed/*`, `models/*`, `experiments/*`, split JSON) is regenerable from the committed code + `data/raw/*`.

## Results summary (all validation-only unless marked "official test")

| model / config | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| mean baseline | 41.94 | 37.50 | −0.00 | 676502 |
| linear | 16.77 | 12.97 | 0.840 | 16032 |
| random forest | 15.67 | 11.14 | 0.860 | 17392 |
| xgboost (w30, cap125) | 14.25 | 10.12 | 0.884 | 13128 |
| lstm (w30, cap125) | 14.21 | 10.91 | 0.885 | 12731 |
| **gru (w30, cap125)** | 13.47 | 9.57 | 0.897 | 19178 |
| tcn (w30, cap125) | 17.80 | 13.26 | 0.819 | 23802 |
| **xgboost @ frozen config (w90, cap45)** | **2.376** | **1.258** | **0.969** | **326** |
| **official test set (evaluated once)** | **2.402** | **1.455** | **0.962** | **14.8** |

Phase 8 ablated: sequence length (15→120), RUL cap (none→45), loss (MSE/MAE/Huber), sensor set (21 vs 15), architecture (4 models) — one factor at a time, validation-only.

## Tests

```bash
.venv\Scripts\python.exe -m pytest        # 49 tests: data contracts, leakage guards,
                                          # golden-file serving check, ablation helpers
```

## Repository layout

- `src/rul_prediction/` — package: `data/` (loader, splitting, preprocessing, sequences), `features/`, `models/` (baselines, xgboost, lstm, gru, tcn), `training/`, `evaluation/` (metrics + NASA score), `serving/`
- `scripts/` — preprocess, run_experiment, build_ablation_table, final_evaluation, serve
- `configs/final_model.yaml` — frozen configuration (commit `02ccbbd`, pre-test)
- `reports/` — phase reports + figures + ablation table
- `notebooks/01_data_exploration.ipynb` — EDA (Phase 2)
- `PROJECT_SPEC.md`, `CHANGELOG.md`, `LICENSE` (MIT)

## License

MIT — see [LICENSE](LICENSE).