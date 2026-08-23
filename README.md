# cmapss-rul-predictive-maintenance

**Leakage-Safe Remaining Useful Life Prediction for Predictive Maintenance Using NASA C-MAPSS**

A reproducible RUL prediction system for turbofan-engine prognostics built with strict leakage safety: engine-level validation splits, training-engine-only preprocessing fitting, fixed pseudo-test manifests for model selection, nested cross-validation with genuinely held-out evaluation engines, and post-hoc evaluation on the official test sets.

> Naming: methodologies were renamed — **M1** (formerly V2), **M2** (formerly V2.1), **M3** (formerly V2.2, current). Older reports and git history may use the old names.

---

## 1. Current methodology — M3

Methodology M3 is the operative protocol (this section). It repairs the M2
declaration (`SUPERSEDED BY METHODOLOGY M3`, see `M3_REPAIR_PLAN.md`):

- the 15 FD001 calibration engines never touch training control — their labels
  are used only AFTER the final model fit, for conformal calibration;
- outer CV folds are genuinely held out: inner early-stop splits
  (58 inner-fit / 10 inner-stop, seeds 4201–4205) control training duration,
  then the model is retrained fixed-duration and evaluated on 17 untouched
  outer-evaluation engines;
- the complete 8-candidate × 5-fold = **40/40** matrix is enforced by a hard
  completeness gate;
- the model-selection policy was **pre-specified** (NASA-primary with a
  pooled-SE RMSE guardrail) in the recorded development session, before any
  M3 result was inspected;
- all training values derive from YAML configs; manifest hashes are canonical
  (platform-independent);
- FD004 variant comparison uses the same clean training-control separation.

## 2. Model-selection design (pre-specified, M3)

Rule locked in `M3_REPAIR_PLAN.md` before the M3 CV ran:

```
PRIMARY : lowest mean NASA per engine (macro over 5 outer folds)
GUARDRAIL: if NASA means differ by less than one pooled standard error
           SE = sqrt((s1^2 + s2^2) / n_folds), prefer lower RMSE
TIE     : smaller |signed bias|
```

Roles are reported separately (they may differ; none is called "best" wholesale):

| role | candidate |
|---|---|
| accuracy champion (lowest CV RMSE) | `lstm_w60_huber` (RMSE 26.19 ± 3.44) |
| NASA-risk champion + **deployment selection** | `xgb_w90_d6` (NASA/engine 368.02 ± 160.40; RMSE 28.35 ± 2.69) |

`experiments/m3/selection_decision.json` holds the full decision; a test
re-derives the selection from the CV summary (no report/config drift).

## 3. Nested engine-level cross-validation

85 development engines, 5 outer engine folds (seed 42): per fold
**68 outer-training / 17 untouched outer-evaluation** engines. Inside the 68,
a deterministic inner split (58 inner-fit / 10 inner-stop, seeds 4201–4205)
controls training duration:

- Stage 1: preprocessing fit on inner-fit only; early stopping monitored on
  inner-stop only → `best_epoch` / `best_iteration`.
- Stage 2: preprocessing refit on all 68 outer-training engines; fresh
  fixed-duration retrain with NO validation data; evaluate the 17 untouched
  outer-evaluation engines on the fixed pseudo-test manifest.

The 17 outer-evaluation engines never influence weights, epochs, iteration
count, early stopping, callbacks, feature fitting or preprocessing.

## 4. Current final FD001 model

Deployment candidate **xgb_w90_d6** (`configs/final_model_m3_fd001.yaml`):

- final `n_estimators = round(median(best_iteration per fold)) + 1 = 92`
  (development-only rule; best iterations 100/91/69/97/53);
- final fit on the **85 development engines only**, no validation data, no
  calibration contact (`scripts/run_m3_freeze.py`);
- canonical engine-ID hashes verified against the split manifest.

| | CV (5 outer folds, mean ± std) | Official FD001 (post-hoc) |
|---|---|---|
| RMSE | 28.35 ± 2.69 | 26.25 |
| MAE | 22.68 | 21.23 |
| R² | 0.735 | 0.601 |
| NASA total | 6,256.4 ± 2,726.8 | 60,963.8 |
| NASA per engine | 368.02 ± 160.40 | 609.64 |

## 5. Uncertainty calibration

Engine-cluster split-conformal, recalibrated on the clean final model
(`scripts/run_m3_conformal.py`): one maximum-error score per held-out
calibration engine across five predefined lifecycle checkpoints
(0.25/0.45/0.65/0.80/0.95) → exactly **15 engine scores**; k = ceil((n+1)(1−α))
clamped; α=0.1 → q = 66.21, α=0.2 → q = 44.80, α=0.3 → q = 41.42.

The interval uses engine-level split-conformal mechanics on 15 engines held
out from M3 fitting and model selection. These engines were inspected during
earlier project iterations, so the interval is an **empirically calibrated
uncertainty interval**, not a pristine one-shot external conformal guarantee.

Formal wording is limited: simultaneous coverage ≥ 1−α holds only under
exchangeability of engines with the predefined checkpoint scheme. Use on
arbitrary uploaded trajectories is an **engineering extrapolation** (labeled
in the app). Post-hoc empirical official coverage at α=0.1: 99% (full-history
engines 100%, short-history 96%).

## 6. Error analysis (post-hoc, descriptive)

`reports/m3_error_analysis.md`: official trajectories are truncated before
failure — `cycle.max()` is observed history, never lifetime. The frozen model
overpredicts systematically (+19.6 cycles mean; 91% of engines), strongest on
short observed histories (< 90 cycles: +29.0 to +49.4). Serving exposes only
the objective `history_is_padded` flag; no empirical risk threshold is applied
(these error patterns are descriptive, not serving triggers).

## 7. Sensitivity analysis (M3 model)

`reports/m3_sensitivity.md` + `reports/tables/m3_*` — sensor occlusion /
counterfactual attribution on the FINAL M3 model (NOT SHAP values):
sensors 4, 11, 3, 9, 12, 7, 20 are the most influential; constant sensors
(1, 5, 10, 16, 18, 19) contribute zero (consistency check); sensor 6, which
M1 flagged, is nearly inert for the M3 model. Conclusions therefore differ
from M1 — reported as measured, no sensor is forced.

## 8. FD004 condition-aware experiment

`scripts/run_m3_fd004.py` — variants A (global scaler), B (global + settings),
C (KMeans k=6 per-regime scalers), D (C + settings + one-hot). Both stages use
the clean protocol: preprocessing fit on inner-fit (150 of 175 training
engines), early stopping on inner-stop (25), stage-2 refit on all 175,
evaluation on the 37 untouched validation engines.

| variant | RMSE | R² | NASA/engine | prediction std |
|---|---|---|---|---|
| A | 72.41 | −0.246 | 75,343 | 0.0 (collapse) |
| B | 72.41 | −0.246 | 75,333 | 0.0 (collapse) |
| **C (selected)** | **29.83** | **0.789** | **1,449** | 70.6 |
| D | 33.97 | 0.726 | 81,963 | 75.9 |

Selection by the same pre-declared principle (NASA per engine, then RMSE):
variant C. Condition-aware normalization still solves the collapse under the
clean validation protocol — and, unlike M2's leaky result, D no longer beats
C on RMSE either. Official FD004 remains **post-hoc** (RMSE 33.66, R² 0.619,
NASA 1,545,798.5). Final FD004 model: `models/m3/fd004_gru_w45_huber_condC.keras`.

## 9. Streamlit demo

`app_m1.py` serves the M3 deployment model (`xgb_w90_d6`) with the
recalibrated interval. Displays model version, predicted raw RUL, observed
cycles, `history_is_padded` + padded timestep count, conformal interval and
calibration method. Includes the engineering-extrapolation disclosure. No OOD
classification anywhere.

## 10. Testing / CI

CI is authoritative for current branch health. Tests are organized into tiers (see `pyproject.toml` / `conftest.py`):

- `static_contract` — repository-integrity, config/provenance/manifest contracts
- `unit` — pure, synthetic, fast
- `tracked_artifacts` — committed `experiments/m3` evidence (falsification)
- `integration` — multi-component, artifact-free
- `app_smoke` — Streamlit import/startup
- `needs_artifacts` — supplemental marker for gitignored data/models

Reproducible commands (Section 15.3):

```bash
.venv\Scripts\python.exe -m pytest -m static_contract
.venv\Scripts\python.exe -m pytest -m unit
.venv\Scripts\python.exe -m pytest -m tracked_artifacts
.venv\Scripts\python.exe -m pytest -m "integration and not needs_artifacts"
.venv\Scripts\python.exe -m pytest -m "app_smoke and not needs_artifacts"
.venv\Scripts\python.exe -m pytest -m "not needs_artifacts"   # aggregate artifact-free guard
.venv\Scripts\python.exe -m pytest -m needs_artifacts --collect-only  # discoverability
.venv\Scripts\python.exe -m pytest          # full local suite (requires gitignored artifacts)
.venv\Scripts\python.exe scripts/check_repository_integrity.py
.venv\Scripts\python.exe scripts/verify_m3_artifacts.py --mode tracked
.venv\Scripts\python.exe scripts/verify_m3_artifacts.py --mode full
```

Historical measurements (e.g., 2026-08-18 snapshot at commit `23cc934` and `6151773`)
are preserved in `CHANGELOG.md` and `reports/repository_integrity_implementation_report.md`
with date, commit, and command context. Do not treat them as permanent current counts;
later commits change collection and counts. See CI workflow (`.github/workflows/ci.yml`)
for the current authoritative check (Python 3.12.10, pinned constraints, `pip check`).

`tests/test_m3_protocol.py` (artifact-free) adds the M3 protocol gates:
calibration isolation (A), outer-fold isolation (B), CV completeness (C),
selection-policy consistency (D), config-driven training (E), canonical
hashing (F), conformal isolation (G), FD004 condition fit-IDs (H).

## 11. Reproduction (M3)

```bash
# manifests + fixed outer/calibration cutoffs + canonical hashes
.venv\Scripts\python.exe scripts\build_m3_manifests.py

# nested CV: 8 candidates x 5 folds (40 runs; completeness gate before summary)
.venv\Scripts\python.exe scripts\run_m3_cv.py

# pre-specified selection -> selection_decision.json + final YAML
.venv\Scripts\python.exe scripts\select_m3_model.py

# clean final fit (85 dev engines, fixed duration, NO validation data)
.venv\Scripts\python.exe scripts\run_m3_freeze.py

# conformal recalibration on 15 held-out calibration engines
# (held out from fitting/selection; inspected in earlier iterations)
.venv\Scripts\python.exe scripts\run_m3_conformal.py

# post-hoc official evaluation (strictly after calibration; falsifies summaries)
.venv\Scripts\python.exe scripts\run_m3_posthoc.py

# FD004: clean A/B/C/D comparison -> config; freeze; post-hoc
.venv\Scripts\python.exe scripts\run_m3_fd004.py
.venv\Scripts\python.exe scripts\run_m3_fd004_freeze.py
.venv\Scripts\python.exe scripts\run_m3_fd004_posthoc.py

# descriptive analyses on the frozen model
.venv\Scripts\python.exe scripts\analyze_m3_errors.py
.venv\Scripts\python.exe scripts\explain_m3_sensitivity.py

# app
.venv\Scripts\python.exe -m streamlit run app_m1.py
```

## 12. Limitations

- Official FD001/FD004 labels are permanently **post-hoc** (inspected in the
  M1-0 audit); no claim of a sealed evaluation.
- Conformal guarantee is limited to exchangeable engines under the predefined
  checkpoint scheme; arbitrary trajectories are an engineering extrapolation.
- The frozen model systematically overpredicts on official engines (post-hoc
  error analysis); the NASA score is sensitive to tail errors.
- FD004 official NASA (1.55M) is much worse than validation NASA — regime
  transfer to the official test remains a limitation.
- No validated OOD detector; serving flags only objective padding
  (`history_is_padded`), no empirical risk flags.

## 13. Historical methodology M2 (superseded)

`SUPERSEDED BY METHODOLOGY M3`. M2 introduced engine-group CV, engine-level
conformal and FD004 condition-aware modeling, but its final FD001 fit used the
15 calibration engines as `validation_data`, its outer folds were not held out
of training control, and its claimed 8×5 matrix was incomplete (rf_w60 had only
fold 1). M2 numbers (e.g. official RMSE 23.04, q = 70.34, official coverage
98%) are historical; see `M2_REPAIR_PLAN.md` and `reports/m2_*.md`.

## 14. Historical methodology M1 (superseded)

`SUPERSEDED BY METHODOLOGY M3`. M1 targeted raw RUL with a single 15-engine
validation set (GRU w45 huber frozen model; official post-hoc RMSE 29.04).
Claims such as "44 engines with lifetime < 128 carry 99.8% of NASA" were
re-interpreted under M2 as observed-history effects. See `AUDIT_M1.md` and
`reports/m1_*.md`.

## 15. Legacy cap-45 experiment (Phase 1–10)

The original experiment (XGBoost @ w90, RUL cap 45) is preserved and labeled a
legacy maintenance-horizon task: `reports/legacy/README.md`,
`configs/legacy_cap45_model.yaml`, `CHANGELOG.md` (Phases 0–10).

## 16. Attribution

- **Dataset:** NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set
  (NASA Ames Prognostics Data Repository, public domain).
- **Origin project:** maintained continuation of the methodology and structure
  of [`aun151214/predictive-maintenance-cmapss`](https://github.com/aun151214/predictive-maintenance-cmapss);
  its cap-45 experiment is preserved as a legacy task; all raw-RUL methodology
  (M1/M2/M3), uncertainty, serving, FD004 work is new. Third-party notices:
  `THIRD_PARTY_NOTICES.md`.

## 17. License

MIT — see [LICENSE](LICENSE).