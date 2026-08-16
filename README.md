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

## Methodology V2.1 (current, replaces V2)

V2.1 repaired four audit findings (`V2_1_REPAIR_PLAN.md`): test trajectories
are truncated before failure, so `cycle.max()` is an **observed history
length**, never a lifetime; model selection now uses **5-fold engine-group CV**
(85 dev / 15 calibration engines) instead of one 15-engine validation set;
conformal calibration is **engine-level** (15 calibration engines, one score
per engine); FD004 is modeled **per operating regime** (KMeans k=6).

| | CV (5-fold, mean±std) | Official FD001 test (post-hoc) |
|---|---|---|
| RMSE | **24.45 ± 3.62** | **23.04** |
| MAE | 16.00 | 15.61 |
| R² | 0.805 | 0.693 |
| NASA total | 5,827.8 ± 4,451.0 | 6,700.59 (V2: 77,387.53) |
| 90% interval | — | engine-cluster q = 70.34; coverage 98% |

Key corrections (details in `reports/v2_1_methodology.md`, `v2_1_cross_validation.md`,
`v2_1_error_analysis.md`, `v2_1_conformal.md`, `v2_1_fd004.md`):

- The V2 claim "44 engines with **lifetime < 128** carry 99.8% of NASA" has no
  lifetime-based counterpart: the implied-lifetime-<128 group is **empty** on
  the official test. The V2 finding was about observed-history length
  (trajectory truncation), re-analyzed in `reports/v2_1_error_analysis.md`.
- **FD004 collapse fixed**: per-regime scaling restores prediction variance
  (RMSE 71.9 → 29.4, R² −0.23 → 0.795, NASA 55× better vs the global-scaler
  baseline; official post-hoc RMSE 33.83). Configs:
  `configs/final_model_v2_1_fd001.yaml`, `configs/final_model_v2_1_fd004.yaml`.

---

## The V2 story in five lines

1. 100 FD001 engines × 21 sensors → padded+masked sliding windows → raw-RUL targets (no cap).
2. Every modeling decision (window, loss, dropout, depth, architecture) was made on a fixed **75-row validation manifest** over a 70/15/15 engine split (seed 42); 45 controlled ablation runs; calibration engines were never used for selection.
3. Winner — **GRU w45 huber** (validation RMSE 13.74, NASA 200.01) — was retrained identically and frozen (`models/v2_frozen_gru_w45_huber.keras`).
4. Error analysis (V2-6) isolated the dominant failure mode, later re-interpreted under V2.1 as an observed-history effect (short observed history overpredicted; "lifetime < 128" was a misnomer — see `reports/v2_1_error_analysis.md`); explainability (V2-7) found late-miss overprediction driven by recent-cycle sensors 2/4/6/7/8; conformal calibration (V2-8) was superseded by engine-level calibration in V2.1 (q = 70.34, official coverage 98%).
5. FD004 generalization (V2-11): the recipe did **not** transfer — the GRU collapsed to a constant under 6 operating conditions. **V2.1 fixed it**: per-regime scaling (variant C) restores variance and cuts NASA 55× (`reports/v2_1_fd004.md`).

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
   │  scripts/analyze_v2_errors.py / explain_v2_sensitivity.py / calibrate_v2_conformal.py
   ▼
reports/v2_error_analysis.md · v2_explainability.md · v2_conformal.md
   │  streamlit run app_v2.py             (serving + intervals + history flags)
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
.venv\Scripts\python.exe scripts\explain_v2_sensitivity.py
.venv\Scripts\python.exe scripts\calibrate_v2_conformal.py

# 6. Serving (Streamlit) — predictions bit-identical to the freeze
.venv\Scripts\python.exe -m streamlit run app_v2.py

# 7. FD004 generalization study (requires FD004 files in data/raw)
.venv\Scripts\python.exe scripts\build_v2_manifests.py --dataset FD004
.venv\Scripts\python.exe scripts\preprocess_v2.py --dataset FD004 --window 45
.venv\Scripts\python.exe scripts\run_v2_fd004.py

# 8. Methodology V2.1: engine-group CV selection, freeze, conformal, errors, FD004 conditions
.venv\Scripts\python.exe scripts\build_v2_1_manifests.py
.venv\Scripts\python.exe scripts\run_v2_1_cv.py            # 8 candidates x 5 folds
.venv\Scripts\python.exe scripts\run_v2_1_freeze.py        # -> models/v2_1/fd001_gru_w45_huber.keras
.venv\Scripts\python.exe scripts\calibrate_v2_1_conformal.py
.venv\Scripts\python.exe scripts\analyze_v2_1_errors.py
.venv\Scripts\python.exe scripts\run_v2_1_fd004.py --variants A B C D
.venv\Scripts\python.exe scripts\run_v2_1_fd004_freeze.py  # -> models/v2_1/fd004_gru_w45_huber_condC.keras
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

### V2.1 — official test sets (post-hoc, current)

| dataset | model | engines | RMSE | MAE | R² | NASA | notes |
|---|---|---|---|---|---|---|---|
| FD001 | gru_w45_huber (85 dev, CV-selected) | 100 | 23.04 | 15.61 | 0.693 | 6,700.59 | engine-cluster interval q=70.34 (α=0.1), official coverage 98% |
| FD004 | gru_w45_huber + per-regime scaling (variant C) | 248 | 33.83 | 22.62 | 0.615 | 1,345,518.43 | fixed the collapse: A baseline RMSE 71.93/R² −0.23/NASA 2.55M |

### V2 — official test sets (post-hoc, historical — superseded by V2.1)

| dataset | engines | RMSE | MAE | R² | NASA | status |
|---|---|---|---|---|---|---|
| FD001 (frozen GRU w45 huber) | 100 | 29.04 | 19.17 | 0.512 | 77,387.53 | post-hoc; "99.8% of NASA from 44 short engines" — superseded: observed-history artifact (see v2_1_error_analysis.md) |
| FD004 (same recipe) | 248 | 64.42 | 54.01 | −0.40 | 2,663,846.31 | **did not transfer** — collapsed to a constant (6 conditions); fixed in V2.1 (variant C) |

### Legacy (Phase 1–10, labeled maintenance-horizon task — RUL cap 45)

| config | RMSE | MAE | R² | NASA |
|---|---|---|---|---|
| **xgboost @ w90 cap45 (frozen)** | **2.376** | **1.258** | **0.969** | **326** |
| official FD001 test (post-hoc) | 2.402 | 1.455 | 0.962 | 14.8 |

Raw-RUL transparency metric of the legacy model: RMSE 50.37, R² −0.47 (predictions capped at 45 by design).

## Uncertainty, explainability, serving (V2-6…V2-9, superseded where noted)

- **Error profile** (`reports/v2_1_error_analysis.md`, supersedes `reports/v2_error_analysis.md`): official test trajectories are truncated before failure — `cycle.max()` is observed history, never lifetime. The implied-lifetime-<128 group is **empty** on the official test. Overprediction concentrates in short observed history (observed 45–127: mean err +17.4, 87.5% of NASA; observed ≥ 128: mean err −4.0, 1.7%).
- **Attribution** (`reports/v2_explainability.md`): exact leave-one-sensor-out attribution (sensitivity/occlusion, not SHAP); sensors 2/4/6/7/8 flip sign (drive late-miss overprediction by +11…+22 cycles); constant sensors contribute zero (consistency check); non-additivity disclosed.
- **Conformal** (`reports/v2_1_conformal.md`, supersedes `reports/v2_conformal.md`): engine-cluster calibration — 15 calibration engines, one max-|error| score per engine, k = ceil((n+1)(1−α)) clamped; α=0.1 → q = 70.34; official coverage 98% (n=100).
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
- `scripts/` — V2: build_v2_manifests, preprocess_v2, run_v2_benchmark/ablation/freeze/fd004, analyze_v2_errors, explain_v2_sensitivity, calibrate_v2_conformal; legacy: preprocess, run_experiment, final_evaluation, serve
- `app_v2.py` — Streamlit serving app
- `configs/` — `final_model.yaml` (legacy frozen config), `legacy_cap45_model.yaml`
- `reports/` — legacy phase reports (labeled), `v2_*.md` phase records, `legacy/README.md`
- `AUDIT_V2.md` — baseline audit (issues + resolutions), `PROJECT_SPEC.md`, `CHANGELOG.md`, `LICENSE` (MIT)

## Acknowledgements

- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set (NASA Ames Prognostics Data Repository, public domain).
- **Origin project:** this repository is a maintained continuation of the methodology and structure of [`aun151214/predictive-maintenance-cmapss`](https://github.com/aun151214/predictive-maintenance-cmapss). Its cap-45 Phase 1–10 experiment is preserved and explicitly labeled as a legacy maintenance-horizon task; all V2 phases (raw-RUL methodology, uncertainty, serving, FD004) are new work by this project.

## License

MIT — see [LICENSE](LICENSE).