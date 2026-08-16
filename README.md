# cmapss-rul-predictive-maintenance

**Leakage-Safe Remaining Useful Life Prediction for Predictive Maintenance Using NASA C-MAPSS**

A reproducible RUL prediction system for turbofan-engine prognostics built phase-by-phase with strict leakage safety: engine-level validation splits, training-engine-only preprocessing fitting, a fixed pseudo-test manifest for model selection, and post-hoc evaluation on the official test sets.

**Primary methodology (V2) targets raw RUL (no cap)** on FD001 and was stress-tested for generalization on FD004. The original Phase 1–10 experiment (XGBoost @ w90, cap 45) is preserved, labeled, and demoted to a **legacy maintenance-horizon task** (`reports/legacy/README.md`).

**V2 frozen model (FD001):** GRU, window 45, huber loss, batch 256, patience 8, seed 42 — selected on a 75-row validation pseudo-test manifest only.

| | Validation (75 rows) | Official FD001 test (post-hoc) |
|---|---|---|
| RMSE | **13.74** | **29.04** |
| MAE | 9.69 | 19.17 |
| R² | 0.877 | 0.512 |
| NASA total | 200.01 | 77,387.53 |

> **Post-hoc status:** the official FD001 labels were inspected during the V2-0 audit, so official-test numbers are labeled **post-hoc**, never "exactly once" (see `AUDIT_V2.md` Issues 1 & 7). The frozen FD001 model is served by a Streamlit app (`app_v2.py`) with conformal intervals.

---

## The V2 story in five lines

1. 100 FD001 engines × 21 sensors → padded+masked sliding windows → raw-RUL targets (no cap).
2. Every modeling decision (window, loss, dropout, depth, architecture) was made on a fixed **75-row validation manifest** over a 70/15/15 engine split (seed 42); 45 controlled ablation runs; calibration engines were never used for selection.
3. Winner — **GRU w45 huber** (validation RMSE 13.74, NASA 200.01) — was retrained identically and frozen (`models/v2_frozen_gru_w45_huber.keras`).
4. Error analysis (V2-6) isolated the dominant failure mode: short test engines (lifetime < 128) carry 99.8% of the NASA error; explainability (V2-7) found late-miss overprediction driven by recent-cycle sensors 2/4/6/7/8; conformal calibration (V2-8) adds honest intervals (90% coverage in-range, ~48% on out-of-distribution short engines — disclosed).
5. FD004 generalization (V2-11): the recipe does **not** transfer to the 6-condition dataset — the GRU collapses to a constant; this negative result is reported, not hidden, and condition-aware modeling is documented as future work.

## Pipeline (V2)

```
data/raw (NASA txt; FD001 + FD004) ── downloaded once, never modified
   │  scripts/build_v2_manifests.py --dataset FD001      (70/15/15 split, seed 42)
   ▼
experiments/splits/<ds>_v2_seed42.json  +  75/75 pseudo-test cutoffs (validation/calibration)
   │  scripts/preprocess_v2.py --dataset FD001 --window 45   (scaler fit on TRAIN engines only)
   ▼
data/processed/v2/FD001/raw/  scaled padded windows (float32) + scaler + metadata
   │  scripts/run_v2_benchmark.py / run_v2_ablation.py  (same 75 validation rows for every model)
   ▼
reports/tables/v2_ablation_results.csv    (validation-only; 45 runs)
   │  scripts/run_v2_freeze.py             (retrain + freeze GRU w45 huber)
   ▼
models/v2_frozen_gru_w45_huber.keras ── post-hoc official FD001 evaluation
   │  scripts/analyze_v2_errors.py / explain_v2_shap.py / calibrate_v2_conformal.py
   ▼
reports/v2_error_analysis.md · v2_explainability.md · v2_conformal.md
   │  streamlit run app_v2.py             (serving + intervals + OOD flag)
   ▼
src/rul_prediction/serving/v2_predictor.py   (golden-tested vs the freeze)
```

## Scientific guardrails (the non-negotiables)

1. Official test data never influences model configuration.
2. Test data never used as `validation_data`.
3. Scalers fitted on training engines only.
4. Windows from one engine never split across partitions.
5. Model selection happens on the fixed validation manifest only; calibration engines are used only for uncertainty calibration (V2-8).
6. Test evaluation happens only after the configuration is frozen; FD001 results are labeled **post-hoc** (labels were inspected in the V2-0 audit).
7. Reproducible seeds (42); metrics always computed programmatically.
8. No fabricated results; failed experiments (e.g., FD004 transfer) are reported, never hidden.
9. Methodology changes documented in `CHANGELOG.md`; per-phase records in `reports/v2_*.md`.

## Reproduce everything (Windows)

```bash
# 0. Environment (Python >= 3.11, tested on 3.12; TensorFlow supports 3.9-3.12)
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e . --no-deps

# 1. FD001: V2 split + pseudo-test manifests (idempotent, seed 42)
.venv\Scripts\python.exe scripts\build_v2_manifests.py --dataset FD001

# 2. FD001: V2 preprocessing (raw RUL, w45; scaler on train engines only)
.venv\Scripts\python.exe scripts\preprocess_v2.py --dataset FD001 --window 45

# 3. Benchmark & ablations (validation-only; 45-run matrix)
.venv\Scripts\python.exe scripts\run_v2_benchmark.py
.venv\Scripts\python.exe scripts\run_v2_ablation.py     # -> reports/tables/v2_ablation_results.csv

# 4. Freeze selected model + post-hoc official evaluation
.venv\Scripts\python.exe scripts\run_v2_freeze.py       # -> models/v2_frozen_gru_w45_huber.keras

# 5. Error analysis / explainability / conformal calibration
.venv\Scripts\python.exe scripts\analyze_v2_errors.py
.venv\Scripts\python.exe scripts\explain_v2_shap.py
.venv\Scripts\python.exe scripts\calibrate_v2_conformal.py

# 6. Serving (Streamlit) — predictions bit-identical to the freeze
.venv\Scripts\python.exe -m streamlit run app_v2.py

# 7. FD004 generalization study (requires FD004 files in data/raw; sealed-labels gate)
.venv\Scripts\python.exe scripts\build_v2_manifests.py --dataset FD004
.venv\Scripts\python.exe scripts\preprocess_v2.py --dataset FD004 --window 45
.venv\Scripts\python.exe scripts\run_v2_fd004.py
```

The legacy Phase 1–10 pipeline (cap-45 XGBoost) is preserved and documented in `reports/legacy/README.md`; its reproduction commands remain in the original `README` history and `CHANGELOG.md` (Phases 0–10).

## Results summary

### V2 — raw RUL, FD001 (validation manifest: 75 fixed rows, seed 42; selected runs)

| model / config | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| mean baseline (constant) | 66.45 | — | — | — |
| rf (w90) | 18.55 | 16.71 | 0.776 | 455.70 |
| xgboost (w90, depth 6) | 19.81 | 17.30 | 0.744 | 531.59 |
| lstm (w60) | 13.98 | 9.98 | 0.873 | 240.21 |
| gru (w60) | 13.89 | 10.97 | 0.874 | 269.34 |
| **gru w45 huber (frozen)** | **13.74** | **9.69** | **0.877** | **200.01** |

Full matrix (45 runs): `reports/tables/v2_ablation_results.csv`.

### V2 — official test sets (post-hoc)

| dataset | engines | RMSE | MAE | R² | NASA | status |
|---|---|---|---|---|---|---|
| FD001 (frozen GRU w45 huber) | 100 | 29.04 | 19.17 | 0.512 | 77,387.53 | post-hoc; 99.8% of NASA from 44 short engines (lifetime < 128) |
| FD004 (same recipe) | 248 | 64.42 | 54.01 | −0.40 | 2,663,846.31 | **does not transfer** — collapses to a constant (6 conditions) |

### Legacy (Phase 1–10, labeled maintenance-horizon task — RUL cap 45)

| config | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| **xgboost @ w90 cap45 (frozen)** | **2.376** | **1.258** | **0.969** | **326** |
| official FD001 test (post-hoc) | 2.402 | 1.455 | 0.962 | 14.8 |

Raw-RUL transparency metric of the legacy model: RMSE 50.37, R² −0.47 (predictions capped at 45 by design).

## Uncertainty, explainability, serving (V2-6…V2-9)

- **Error profile** (`reports/v2_error_analysis.md`): training lifetime min = 128; official engines below it (44/100) are missed late 80% of the time and carry 99.8% of the NASA score; in-range engines: mean error −4.3 cycles.
- **Attribution** (`reports/v2_explainability.md`): exact leave-one-sensor-out attribution; sensors 2/4/6/7/8 flip sign (drive late-miss overprediction by +11…+22 cycles); constant sensors contribute zero (consistency check); non-additivity disclosed.
- **Conformal** (`reports/v2_conformal.md`): split-conformal calibration on 75 calibration rows; 90% interval width 24.1 cycles; coverage on official test 69% overall, 85.7% in-range vs 47.7% for short engines — the OOD degradation is quantified, not hidden.
- **Serving** (`app_v2.py`): predictions bit-identical to the freeze (golden-tested), 90% intervals, alarm lower bound, OOD flag.

## Tests

```bash
.venv\Scripts\python.exe -m pytest        # 93 tests (49 legacy + 44 V2: manifests,
                                          # preprocessing, features, conformal, serving)
.venv\Scripts\python.exe -m pytest -m "not needs_artifacts"   # CI subset: 78 tests,
                                          # no data/model artifacts required
```

CI (GitHub Actions, `.github/workflows/ci.yml`) runs the artifact-free subset on Python 3.12.

## Repository layout

- `src/rul_prediction/` — package: `data/` (loader, splitting, V2 manifests, windows, v2 preprocessing), `benchmark/v2.py` (shared runner), `evaluation/` (metrics, NASA score, conformal), `models/`, `features/`, `training/`, `serving/` (legacy + `v2_predictor.py`)
- `scripts/` — V2: build_v2_manifests, preprocess_v2, run_v2_benchmark/ablation/freeze/fd004, analyze_v2_errors, explain_v2_shap, calibrate_v2_conformal; legacy: preprocess, run_experiment, final_evaluation, serve
- `app_v2.py` — Streamlit serving app
- `configs/` — `final_model.yaml` (legacy frozen config), `legacy_cap45_model.yaml`
- `reports/` — legacy phase reports (labeled), `v2_*.md` phase records, `legacy/README.md`
- `AUDIT_V2.md` — baseline audit (issues + resolutions), `PROJECT_SPEC.md`, `CHANGELOG.md`, `LICENSE` (MIT)

## Acknowledgements

- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set (NASA Ames Prognostics Data Repository, public domain).
- **Origin project:** this repository is a maintained continuation of the methodology and structure of [`aun151214/predictive-maintenance-cmapss`](https://github.com/aun151214/predictive-maintenance-cmapss). Its cap-45 Phase 1–10 experiment is preserved and explicitly labeled as a legacy maintenance-horizon task; all V2 phases (raw-RUL methodology, uncertainty, serving, FD004) are new work by this project.

## License

MIT — see [LICENSE](LICENSE).