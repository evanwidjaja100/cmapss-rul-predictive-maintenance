# PROJECT_SPEC.md

Leakage-Safe Remaining Useful Life Prediction for Predictive Maintenance Using NASA C-MAPSS

## 1. Problem

Predict the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor time series using the NASA C-MAPSS dataset. RUL is defined as the number of operational cycles remaining before engine failure. Primary target: **raw RUL (no cap)**.

## 2. Dataset

- Initial dataset: **FD001** (single operating condition, one fault mode, 100 training / 100 test engines)
- Second dataset: **FD004** (multiple operating regimes, 249 training / 248 test engines; condition-aware experiment)
- Raw files (FD001): `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`
- Raw files (FD004): `train_FD004.txt`, `test_FD004.txt`, `RUL_FD004.txt`

## 3. Validation methodology — CURRENT: Methodology M3

Engine-level partitions (FD001): **15 calibration engines** (reserved for
uncertainty calibration ONLY within M3 — their labels are first touched by
M3 after the final model fit, during conformal calibration; they were also
inspected in earlier M1/M2 iterations) and **85 development engines**.

**Nested engine-group cross-validation (M3):**

- 5 outer engine folds (seed 42): per fold **68 outer-training** engines and
  **17 untouched outer-evaluation** engines;
- inside the 68, a deterministic inner early-stop split — `random.Random(4200 +
  fold)` on sorted IDs, **58 inner-fit / 10 inner-stop**, seeds 4201–4205
  (documented before any M3 result was inspected);
- Stage 1: preprocessing fit on inner-fit ONLY; early stopping / best iteration
  monitored on inner-stop ONLY;
- Stage 2: preprocessing refit on ALL 68 outer-training engines; fresh
  fixed-duration retrain (best_epoch, or best_iteration + 1 for XGBoost) with
  NO validation data;
- evaluation of the 17 untouched outer-evaluation engines on the fixed outer
  pseudo-test manifest (5 balanced lifecycle cutoffs 0.25/0.45/0.65/0.80/0.95
  per engine);
- the outer-evaluation engines never influence weights, epoch count, best
  iteration, early stopping, callbacks, feature fitting or preprocessing.

**Final FD001 fit (M3):** on the 85 development engines only, fixed duration
(`final_epoch_count = round(median(best_epoch))` for deep models;
`final_n_estimators = round(median(best_iteration)) + 1` for XGBoost), with
NO validation_data and NO calibration contact. Conformal calibration happens
afterwards on the 15 calibration engines.

**CV completeness:** the summary is only generated when every declared
candidate has folds {1..5} and the matrix has exactly 8 × 5 = 40 rows
(`assert_cv_complete`).

**FD004 (M3):** 175 training / 37 validation / 37 calibration engines;
inner split 150 inner-fit / 25 inner-stop (seed 4201); variant comparison uses
the same two-stage protocol; condition models (KMeans, cluster scalers,
settings scaler) fit only on permitted training rows at each stage.

The **official test sets are used for final post-hoc evaluation only** and are
permanently labeled POST-HOC. They never influence model selection, training
control, preprocessing fitting or configuration.

## 4. Metrics

- RMSE
- MAE
- R²
- NASA / PHM asymmetric score (also normalized per engine)

All metrics are computed programmatically from predictions; no metric is
fabricated. Headline numbers are independently recomputed from saved
prediction CSVs (falsification pass).

## 5. Non-negotiable rules

1. Official test data cannot influence model configuration (post-hoc only).
2. Calibration engines cannot enter `model.fit()`, `validation_data`,
   early stopping, epoch/iteration selection or preprocessing fitting.
3. Outer-evaluation engines cannot enter training control (nested CV).
4. Scalers / condition models are fitted using permitted training rows only.
5. Sequence windows from the same engine cannot cross partitions.
6. Official test results are generated only after: CV complete → selection
   policy applied → YAML frozen → final dev-only fit → conformal calibration.
7. Experiments use reproducible random seeds (global 42; inner seeds 4201–4205).
8. Metrics are always generated programmatically.
9. Results must never be fabricated; failed experiments are reported.
10. Any methodology change becomes a new documented experimental cycle.

## 6. Environment

- Python: 3.12.10 (pinned in CI and M3 configs; local venvs must match).
- All libraries are installed inside `<project>/.venv`. Global pip is not used.

## 7. Roadmap

| Phase | Scope |
|---|---|
| 0 | Project initialization and methodology |
| 1 | Data acquisition and validation |
| 2 | Exploratory data analysis |
| 3 | Leakage-safe data splitting |
| 4 | RUL preprocessing and sequence generation |
| 5 | Classical ML baselines |
| 6 | LSTM and GRU baselines |
| 7 | TCN improvement model |
| 8 | Ablation and hyperparameter experiments |
| 9 | Freeze configuration and final test |
| 10 | Engineering error analysis |
| 11 | Explainable AI |
| 12 | Prediction uncertainty |
| 13 | Streamlit dashboard |
| 14 | Testing, reproducibility and CI |
| 15 | FD004 generalization study |
| 16 | Final README and portfolio presentation |
| — | M1 / M2 / M3 methodology cycles (see below) |

## 8. Reproducibility

Every M3 run records: git commit, dataset, candidate, outer fold, inner
split seed, engine IDs + canonical hashes, window, features, preprocessing,
model hyperparameters, best epoch / best iteration, final retraining duration,
training time, RMSE/MAE/R²/NASA/signed bias, software versions
(`experiments/m3/fd001_outer_metadata.jsonl`). Manifest hashes are canonical
(platform-independent; `src/rul_prediction/data/canonical_hash.py`).
Default deterministic seed: **42**; inner split seeds 4201–4205.

## 9. Governance

- Each methodology cycle ends with tests, acceptance-criteria verification, a
  commit, and a STOP.
- Failed experiments are reported, never hidden. Test performance is reported
  honestly even when disappointing.

## 10. Methodology M3 — current execution status

Repairs `SUPERSEDED BY METHODOLOGY M3` (see `M3_REPAIR_PLAN.md`):

| Item | M3 state |
|---|---|
| final FD001 fit | 85 dev engines only, fixed duration, no calibration contact (`run_m3_freeze.py`) |
| outer folds | genuinely held out (inner 58/10 splits control duration; 17-eval engines untouched) |
| CV matrix | complete **40/40** (8 candidates × 5 folds; hard completeness gate) |
| selection policy | pre-specified (NASA per engine primary; pooled-SE RMSE guardrail); `selection_decision.json`; accuracy champion `lstm_w60_huber`, NASA-risk champion + deployment `xgb_w90_d6` |
| conformal | recalibrated on 15 held-out calibration engines (inspected in earlier iterations → empirically calibrated, not pristine); q(0.1)=66.21; formal wording limited; engineering extrapolation labeled |
| FD004 | clean two-stage A/B/C/D comparison; variant C selected (NASA 1,449/engine, RMSE 29.83); collapse still fixed; post-hoc official RMSE 33.66 |
| sensitivity | rerun on the M3 model (occlusion; sensors 4/11/3/9/12/7/20 most influential) |
| configs | `configs/final_model_m3_fd001.yaml`, `configs/final_model_m3_fd004.yaml` drive the freeze scripts |
| tests | CI-verified artifact-free tiers (`static_contract`, `unit`, `tracked_artifacts`, `integration`, `app_smoke`; aggregate `not needs_artifacts` guard); historical measurements are dated snapshots in the implementation report, not permanent counts; `tests/test_m3_protocol.py` (Tests A–H) |
| Streamlit | serves M3 model + recalibrated interval + disclosure (`app_m1.py`) |

## 11. Methodology M2 — historical (superseded)

`SUPERSEDED BY METHODOLOGY M3`. M2 introduced engine-group CV
(85 dev / 15 calibration), engine-level conformal and FD004 condition-aware
modeling, but the final FD001 fit passed calibration engines as
`validation_data`, outer folds were not held out of training control, and the
claimed 8×5 matrix was incomplete (rf_w60: fold 1 only). Historical numbers
(e.g. official FD001 RMSE 23.04, q = 70.34) are recorded in `reports/m2_*.md`.

## 12. Methodology M1 — execution status (historical)

Methodology M1 (Phases M1-0 … M1-12) re-based the project on a raw-RUL target
and a fixed pseudo-test manifest, with a 70/15/15 engine split and a single
15-engine validation set. `SUPERSEDED BY METHODOLOGY M3`.

| Roadmap phase | M1 execution | Status |
|---|---|---|
| 10 error analysis | M1-6 (`reports/m1_error_analysis.md`) | done (historical) |
| 11 explainable AI | M1-7 (`reports/m1_explainability.md`) | done (historical; SHAP-free leave-one-sensor-out) |
| 12 prediction uncertainty | M1-8 (`reports/m1_conformal.md`) | done (historical; q = 24.10 superseded) |
| 13 Streamlit dashboard | M1-9 (`app_m1.py`, `reports/m1_serving.md`) | done (serves M3 now) |
| 14 testing / CI | M1-10 (`tests/`, `.github/workflows/ci.yml`) | done (historical counts preserved with date/commit; current health via CI tiers and reproducible commands, not permanent numeric claims) |
| 15 FD004 generalization | M1-11 (`reports/m1_fd004.md`) | done — negative result: recipe collapsed under 6 operating conditions; condition-aware modeling implemented in M2/M3 |
| 16 final docs | M1-12 (`README.md`, `CHANGELOG.md`) | done (historical) |

Key M1 facts (historical):

- Primary target = **raw RUL**; the legacy cap-45 XGBoost experiment is
  preserved as a labeled maintenance-horizon task (`reports/legacy/README.md`).
- M1 split: 70 train / 15 validation / 15 calibration engines, seed 42.
- Frozen M1 model: GRU w45 huber — official FD001 post-hoc RMSE 29.04 / NASA
  77,387.53.
- All official FD001 results are labeled **post-hoc** (labels inspected in the
  M1-0 audit; see `AUDIT_M1.md` Issues 1 & 7).
- FD004 official labels are post-hoc by policy from M2 on (never sealed).