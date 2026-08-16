# CHANGELOG

## Methodology V2.1 (repair) — 2026-08-15 (`67a0e58` + follow-ups)

> V2.1 repairs four audit findings (`V2_1_REPAIR_PLAN.md`, R1–R20). V2
> conclusions that are superseded are labeled `SUPERSEDED BY METHODOLOGY V2.1`
> in their reports; V2 remains historical record.

- **Lifetime semantics corrected (R1/R2/R3/R4):** official test trajectories
  are truncated before failure, so `cycle.max()` is observed history, never
  lifetime. `implied_failure_cycle = observed_cycles + true_rul` is the
  labeled quantity. The V2 claim "44 engines with lifetime < 128 carry 99.8%
  of NASA" has **no lifetime-based counterpart — that group is empty on the
  official test**; it was an observed-history artifact. Serving no longer
  classifies OOD: `history_is_padded` (objective) + empirical short-history
  risk flag. `reports/v2_1_error_analysis.md`.
- **Engine-group 5-fold CV selection (R5/R6/R7/R8):** 85 development / 15
  calibration engines (V2 calibration IDs preserved), 5-fold group CV
  (seed 42), balanced fractions 0.25/0.45/0.65/0.80/0.95 fixed pre-comparison,
  8 bounded candidates. Winner **gru_w45_huber** (RMSE 24.45±3.62, NASA
  5,828±4,451, bias −0.25). Frozen on 85 dev engines: official post-hoc
  RMSE 23.04 / R² 0.693 / NASA 6,700.59 (11.5× vs V2's 77,387.53).
  `configs/final_model_v2_1_fd001.yaml`, `reports/v2_1_cross_validation.md`.
- **Engine-cluster conformal (R9/R10/R11):** one max-|error| score per
  calibration engine over its 5 checkpoints (n=15), k = ceil((n+1)(1−α))
  clamped; α=0.1 → q = 70.34; official coverage 98%. V2-8's n=75 row-level
  q=24.10 superseded. `reports/v2_1_conformal.md`.
- **FD004 condition-aware modeling (R13/R14):** A/B (global scaler ± settings)
  collapse reproduced (pred std 0.0, RMSE ~72); **C (KMeans k=6 per-regime
  scalers) restores variance** — RMSE 29.37, R² 0.795, NASA 46,165 (55× vs A);
  D (C + one-hot) RMSE-optimal (27.22) but NASA 68,981 — not selected. Frozen
  C official post-hoc: RMSE 33.83 / NASA 1,345,518 (1.9× better than the
  V2-11 collapse). FD004 official = post-hoc by policy from V2.1 on.
  `configs/final_model_v2_1_fd004.yaml`, `reports/v2_1_fd004.md`.
- **Engineering (R15–R20):** `explain_v2_shap.py` renamed
  `explain_v2_sensitivity.py` (occlusion, not SHAP); `pyproject.toml`
  `requires-python = ">=3.11,<3.13"`; `THIRD_PARTY_NOTICES.md`; CI markers
  declared; `experiments/splits/` + `experiments/v2_1/` un-ignored for
  auditability; V2 reports bannered SUPERSEDED; README/PROJECT_SPEC updated.
- Tests: 110 → **115 passing** (16 V2.1 methodology + 5 FD004 condition +
  rewritten serving tests); app smoke `python -c "import app_v2"` passes.

## Methodology V2 (Phases V2-0 … V2-12) — 2026-08-15

> **Correction (V2-12):** legacy entries below claim the official test set was
> "evaluated exactly once". The official FD001 labels were re-inspected during
> the V2 audits, so from V2 onward all official FD001 results are labeled
> **post-hoc** (see `AUDIT_V2.md` Issues 1 & 7).

### V2-0 — Baseline audit (`c3c8694`)

- `AUDIT_V2.md`: 11 confirmed issues (capped-target headline, "exactly once" claims, missing attribution, absolute lock path, skipped roadmap phases); legacy Phase 1–10 experiment labeled (`configs/legacy_cap45_model.yaml`, `reports/legacy/README.md`); no code or results modified.

### V2-1 — Methodology setup (`2aedc42`)

- V2 plan: primary target = **raw RUL** (no cap); 70/15/15 engine split (seed 42); fixed pseudo-test manifests (75 validation + 75 calibration rows, 5 lifecycle fractions); shared runner `src/rul_prediction/benchmark/v2.py`; consistent padded+masked history for train and inference.

### V2-2 — V2 preprocessing (`2c93185`)

- `src/rul_prediction/data/v2_preprocessing.py` + `scripts/preprocess_v2.py`: raw-RUL targets, scaler fit on train engines only, artifacts under `data/processed/v2/FD001/raw/`.

### V2-3 — Raw-RUL benchmark (`e88388b`)

- mean/linear/rf/xgboost/lstm/gru/tcn on the fixed 75-row validation manifest; recurrent models dominate (gru w30: RMSE 24.66); linear regression unusable on raw RUL.

### V2-4 — Window & hyperparameter ablations (`67664ce`)

- 45 runs (windows 15–90, loss, dropout, depth); **selection: GRU w45 huber — validation RMSE 13.74, MAE 9.69, R² 0.877, NASA 200.01**. `reports/v2_ablation.md`, `reports/tables/v2_ablation_results.csv`.

### V2-5 — Freeze + post-hoc FD001 (`be9252a`)

- `models/v2_frozen_gru_w45_huber.keras`; validation reproduced bit-exact (13.7406); official FD001 post-hoc: RMSE 29.0377, MAE 19.1715, R² 0.5117, NASA 77,387.53; 5 late misses = 96.6% of NASA. `reports/v2_freeze.md`.

### V2-6 — Error analysis (`b3742bc`)

- Training lifetime min 128; official engines below it (44/100) are missed late 80% of the time and carry 99.8% of NASA; in-range engines: mean error −4.3 cycles. `reports/v2_error_analysis.md`.

### V2-7 — Explainability (`74b4ce2`)

- shap 0.52.0 (keras 3.15.1 breaks Deep/Gradient/Kernel explainers — documented); exact leave-one-sensor-out attribution; sensors 2/4/6/7/8 flip sign (late-miss overprediction); constant sensors ≈ 0 attribution; non-additivity disclosed. `reports/v2_explainability.md`.

### V2-8 — Conformal uncertainty (`3d3f644`)

- Split-conformal calibration on the 75 calibration rows; 90% interval width 24.10 cycles; coverage: calibration 92%, validation 88%, official 69% (85.7% in-range vs 47.7% OOD); lower-bound predictor cuts official NASA 3.2× at α=0.2. `reports/v2_conformal.md`.

### V2-9 — Streamlit serving (`815e718`)

- `app_v2.py` + `src/rul_prediction/serving/v2_predictor.py` (predictions bit-identical to the freeze, golden-tested); 90% conformal intervals, alarm lower bound, OOD flag; `reports/v2_serving.md`.

### V2-10 — CI / dependency cleanup (`978b060`)

- `requirements-lock.txt` regenerated (path-free editable line; includes shap/streamlit); Python version policy reconciled (>= 3.11, tested on 3.12); `needs_artifacts` marker registered; GitHub Actions workflow runs the artifact-free subset (78 passed, verified on an artifact-free tree).

### V2-11 — FD004 generalization study (`27a3e64`)

- FD004 acquired (Kaggle mirror; NASA S3 403 for all probed paths); **sealed-labels gate**: repo-wide grep proved no code reads `RUL_FD004.txt`, sha256 pinned at download and re-verified before the first-ever label read. The exact GRU recipe **does not transfer**: collapses to a constant (official RMSE 64.42, R² −0.40, NASA 2,663,846.31; validation R² −2.23) under 6 operating conditions. Condition-aware preprocessing documented as future work. `reports/v2_fd004.md`.

### V2-12 — Final documentation & attribution

- README rewritten: raw-RUL headline with post-hoc labeling, V2 pipeline/reproduction, FD004 verdict, attribution (NASA C-MAPSS + `aun151214/predictive-maintenance-cmapss`).
- "Exactly once" claims corrected (labeled historical/post-hoc) in `reports/phase9_final_evaluation.md`, `reports/phase10_serving.md`, `configs/final_model.yaml`, `scripts/final_evaluation.py`.
- Mojibake check on legacy reports: verified clean (valid UTF-8); `reports/phase8_ablation.md` "6918 engines" typo → "sequences/windows"; notebook absolute path made machine-neutral.
- PROJECT_SPEC.md: V2 status section mapping roadmap phases 10–16.

## [0.11.0] — Phase 10 — 2026-08-15

### Added
- `src/rul_prediction/serving/inference.py`: `RulPredictor` — loads frozen config (validated: model/variant/window/max_rul must match), Phase 9 model + train-only scaler; predicts per-unit RUL with the exact Phase 9 pipeline (scale, 90-cycle windows with zero-padding for short units, 169 features, clip [0,45]).
- `scripts/serve.py`: `serve` (stdlib HTTP, `GET /health`, `POST /predict` with positional or named C-MAPSS rows) and `batch` (predicts a test-style file, optional RUL metrics).
- `tests/test_inference_golden.py`: golden-file check — serving predictions must match the Phase 9 official test predictions per unit (<=1e-4); short-unit padding check.

### Verification
- Golden test passes; HTTP smoke test: 100 units / 26 padded on the full test payload.
- Batch mode reproduces Phase 9 official metrics exactly: RMSE 2.4024 | MAE 1.4548 | R² 0.9619 | NASA 14.767.
- 49 pytest tests pass. No retraining; model loaded, never refit.

### Notes
- Only stdlib for the HTTP layer (no FastAPI/uvicorn dependency); auth/TLS/Docker intentionally skipped until a real deployment target exists.
- Cosmetic xgboost warning when loading the `.joblib`-named UBJSON model artifact.

## [0.10.0] — Phase 9 — 2026-08-15

### Added
- `configs/final_model.yaml`: configuration frozen BEFORE test contact (Phase 8 winner: XGBoost @ w90_c45_all — window 90, RUL cap 45, all sensors, seed 42) with its pre-test validation metrics for reproducibility gating.
- `scripts/final_evaluation.py`: two-stage harness — (1) reproducibility pass on validation engines that must reproduce the frozen metrics within 1e-3 (else abort before test), (2) ONE-TIME official test evaluation writing `experiments/FD001_final_test_results.json` + per-unit `test_predictions.csv` (with `padded_short` flags) and `models/final/FD001_final_model.joblib`.
- `tests/test_final_evaluation.py`: window-padding logic (short units zero-padded at start, tail preserved; long units take last window).

### Official test-set results (FD001, 100 units, evaluated exactly once)
- Clipped at frozen cap 45 (primary convention): **RMSE 2.402 | MAE 1.455 | R² 0.962 | NASA 14.77**.
- Raw RUL (transparency only; predictions capped at 45 by design): RMSE 50.37, R² −0.47.
- 26/100 test units shorter than the 90-cycle window were left-padded (documented in predictions CSV).

### Notes
- Full audit trail in `reports/phase9_final_evaluation.md`; commit `02ccbbd` froze config + harness before the test run.
- 47 pytest tests pass.

## [0.9.0] — Phase 8 — 2026-08-15

### Added
- `scripts/preprocess.py` variant support: `--window`, `--max-rul` (or `none`), `--sensors all|varying`; each (window, cap, sensors) combination is built into `data/processed/<dataset>_w<W>_c<cap>_<sensors>/` with its own scaler (fit on TRAINING ENGINES ONLY), sequences, and metadata JSON. Constant-sensor detection runs on the training partition only.
- `scripts/run_experiment.py --variant ...`: window and RUL cap now come from the variant (X shapes / `c<cap>` parsing); deep-model notes carry the custom `--notes` text; results.csv `RUL cap` column reflects the variant (incl. `none`).
- `scripts/build_ablation_table.py`: derives `reports/tables/ablation_results.csv` from validation-only `experiments/results.csv` by classifying each row into factors A-D/E.
- `tests/test_experiment_helpers.py`; `tests/test_artifacts.py` extended with a varying-sensor variant guard (15 features, still scaled).

### Ablation findings (validation engines only, seed 42; official test set untouched)
- **A window**: 90 wins (RMSE 11.17 vs 13.47 at 30); 120 slightly worse (11.68) as sequences thin out.
- **B RUL cap**: tighter is better down to **45** (RMSE 2.63, R² 0.950); verified the gain is real, not a constant-predictor artifact (per-bucket tracking corr 0.979, 0% of predictions stuck at the cap).
- **C loss**: MSE beats MAE (15.10) and Huber (15.38).
- **D sensors**: keeping all 21 beats dropping the 6 constants for GRU (13.47 vs 14.23); XGBoost barely prefers 15.
- **E architecture @ final variant (w90_c45_all)**: **XGBoost wins** (RMSE 2.376, R² 0.969, NASA 326) over GRU (2.654) and LSTM (2.922) — a flip from the base config where GRU was champion; documented honestly.
- Recurring negative: with a tight cap the last-45-cycle signal favors hand-built window features over learned recurrence.

### Locked for Phase 9
- Final config (validation-selected): **XGBoost @ variant w90_c45_all** (window 90, RUL cap 45, all sensors, seed 42) → RMSE 2.376 | MAE 1.258 | R² 0.969 | NASA 326. Composition check w90_c45 GRU = 2.65 (no factor interaction loss).
- Full table: `reports/tables/ablation_results.csv`; details in `reports/phase8_ablation.md`.

### Notes
- `data/processed` now holds 11 variants (ignored by git); metadata JSON records scaler fit partition, removed sensors, sequence counts.
- 45 pytest tests pass.

## [0.8.0] — Phase 7 — 2026-08-14

### Added
- `src/rul_prediction/models/tcn.py`: TCN with causal convolutions, 4 residual dilated blocks (dilations 1, 2, 4, 8, kernel 3, receptive field 61 >= window 30), BatchNorm, dropout 0.2, global pooling.
- `scripts/run_experiment.py`: `--model tcn` with identical protocol to LSTM/GRU (same scaled 30x21 windows, MSE, Adam 1e-3 clipnorm=1, batch 128, patience 8, seed 42).

### Comparison table (validation engines only, seed 42, RUL cap 125)
| model | RMSE | MAE | R2 | NASA |
|---|---|---|---|---|
| xgboost | 14.25 | 10.12 | 0.884 | 13128 |
| lstm | 14.21 | 10.91 | 0.885 | 12731 |
| **gru** | **13.47** | **9.57** | **0.897** | 19178 |
| tcn | 17.80 | 13.26 | 0.819 | 23802 |

**Research question answer (this configuration):** a causal-convolutional TCN does NOT outperform the recurrent architectures on FD001 validation; GRU remains the validation champion. TCN still beats linear/logistic-style engineered baselines but with the worst NASA score of the four contenders. The residual dilated block assumption (that long sparse-receptive-field temporal features add value here) did not materialize — candidates for Phase 8 ablations: filters, dilation depth, pooling, dropout.

### Notes
- TCN: 123,969 parameters; early-stopped with best-weights restore (logged in results.csv).
- Guardrails unchanged: engine-disjoint validation, official test untouched.

### Added
- `src/rul_prediction/models/lstm.py` and `gru.py`: comparable recurrent baselines (128 -> Dropout 0.3 -> 64 -> Dropout 0.3 -> Dense 32 -> Dense 1; Adam with clipnorm=1.0).
- `src/rul_prediction/training/trainer.py` (`set_seed`, `train_sequence_model`) and `callbacks.py` (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint). Validation partition is engine-disjoint; official test data never used.
- `scripts/run_experiment.py` extended with `--model lstm|gru`, `--loss`, `--epochs`, `--batch-size`, `--patience`, `--learning-rate`.
- `tests/test_artifacts.py`: data-contract guards (windows scaled, targets clipped).
- Installed `tensorflow` 2.21.0 (project-local); regenerated `requirements-lock.txt`.

### CRITICAL BUG FIX (data preprocessing)
Found and fixed a silent Phase-4 defect: `scripts/preprocess.py` computed the scaled feature matrix but wrote RAW sensor values into the persisted sequence windows (`transform()` result was discarded at build time). Symptoms: every NN model collapsed to a constant RUL prediction (RMSE ~41.9, pred_std=0.0) while classical models were unaffected (per-column affine rescaling is absorbed by linear/trees). Diagnosis via bisection (toy task learned fine; fabricated-target LSTM learned fine; real-input/real-target constant) then direct artifact inspection (sensor_9 raw ~9064, sensor_1 constant 518.67). Fix: scale features in place before `make_sequences`, regenerated all `data/processed` artifacts, added `test_artifacts.py` guards. Sequence counts unchanged (14022 / 3709).

### Validation benchmark (validation engines only, seed 42; RUL cap 125) - AFTER fix
| model | RMSE | MAE | R2 | NASA |
|---|---|---|---|---|
| mean | 41.94 | 37.50 | -0.00 | 676502 |
| linear | 16.77 | 12.97 | 0.840 | 16032 |
| random forest | 15.67 | 11.14 | 0.860 | 17392 |
| xgboost | 14.25 | 10.12 | 0.884 | 13128 |
| lstm | 14.21 | 10.91 | 0.885 | 12731 |
| **gru** | **13.47** | **9.57** | **0.897** | 19178 |

**Validation champion: GRU** (best RMSE/MAE/R2). LSTM has best NASA score (its error profile is less late-biased - analysed in Phase 10). Official test data not consulted.

### Notes
- LSTM best epoch 6, GRU stopped early via patience (epochs logged in `experiments/results.csv` notes).
- Model weights checkpointed under `models/checkpoints/` (gitignored).

### Added
- `src/rul_prediction/features/engineered_features.py`: history-only window features (last value, mean, std, min, max, linear slope, last-5/last-10 means) per sensor + engine age — never uses future cycles.
- `src/rul_prediction/models/baseline.py`: `MeanBaseline`, `linear_regressor`, `random_forest`.
- `src/rul_prediction/models/xgboost_model.py`: `xgboost_regressor` with early stopping on the validation partition.
- `src/rul_prediction/evaluation/metrics.py` (RMSE, MAE, R2) and `nasa_score.py` (PHM asymmetric score).
- `scripts/run_experiment.py`: trains a model on engineered features, evaluates on validation engines only, appends to `experiments/results.csv`.
- Tests: `tests/test_metrics.py`, `tests/test_features.py`.
- Installed `xgboost` into `.venv`; regenerated `requirements-lock.txt`.

### Classical benchmark (validation engines only, seed 42, 169 engineered features, RUL cap 125)
| model | RMSE | MAE | R2 | NASA |
|---|---|---|---|---|
| mean | 41.94 | 37.50 | -0.00 | 676502 |
| linear | 16.81 | 13.02 | 0.839 | 16076 |
| random forest | 15.67 | 11.16 | 0.860 | 17421 |
| **xgboost** | **14.23** | **10.14** | **0.885** | **13271** |

**Validation champion: XGBoost** (best on all four metrics). Official test data not consulted.

### Notes
- NASA score favors early over late predictions; all models here tend to late-predict at validation, keeping NASA scores high (error analysis in Phase 10).

### Added
- `src/rul_prediction/data/preprocessing.py`: `add_rul` (max_cycle - cycle, optional clip), `fit_scaler` / `transform` / `save_scaler` / `load_scaler` (scaler fitted on training engines only).
- `src/rul_prediction/data/sequences.py`: `make_sequences` (sliding windows built per engine, never crossing engine boundaries; target = RUL of final cycle in window).
- `src/rul_prediction/data/splitting.py`: added `read_split_file` to consume the pinned split.
- `scripts/preprocess.py`: full processing path (builds scaled windows + persisted artifacts); `--validate-only` retained.
- Tests: `tests/test_sequences.py` (RUL, clipping, dims/dtype, target correctness, short-engine skip, boundary integrity, scaler-fitted-on-train-only, train/validation disjointness).
- Installed `scikit-learn`, `joblib` into `.venv` (also brought in `scipy`, resolving the earlier missing-dependency warning). Regenerated `requirements-lock.txt`.

### Products (FD001, generated, gitignored under `data/processed/`)
- `FD001_scaler.joblib` (fitted on 80 training engines only)
- `FD001_train_sequences.npz`: 14022 windows x (30, 21)
- `FD001_validation_sequences.npz`: 3709 windows
- `FD001_scaled_features.npz`: scaled train/validation/test feature arrays + engine IDs
- `FD001_metadata.json`

### Diagnostic output (real)
Train engines: 80 | Validation engines: 20 | Overlap: 0 | Train sequences: 14022 | Validation sequences: 3709 | Sequence length: 30 | Input features: 21 | Scaler fit partition: TRAIN ONLY
(Sequence counts independently recomputed and match.)

### Notes
- Clip cap `max_rul=125` and window `30` are Phase-4 defaults; final values selected in Phase 8 using validation only. Constant sensors retained in the 21 features (Phase 8 ablation).

### Added
- `src/rul_prediction/data/splitting.py`: deterministic engine-level train/validation split (seed 42, 80/20) with a `python -m` CLI.
- `experiments/splits/FD001_seed42.json`: pinned 80/20 partition (80 train / 20 validation engines, zero overlap).
- Tests: `tests/test_splitting.py` (determinism, no overlap, full coverage, ratio, seed sensitivity, JSON round-trip).

### Notes
- Split performed on engine IDs only; overlapping windows are never produced here (Phase 4 consumes this split).

### Added
- `notebooks/01_data_exploration.ipynb`: fully executed EDA notebook using project package functions (self-anchors to repo root; 25 cells, 0 errors).
- Figures saved to `reports/figures/eda/`: lifetime distribution, sensor variance, sensor trajectories, sensor-RUL correlation, sensor-sensor heatmap.
- Installed `matplotlib`, `jupyterlab`, `ipykernel` into `.venv`; project-local kernelspec registered at `.venv/share/jupyter/kernels/python3` (no global Jupyter config).
- Regenerated `requirements-lock.txt`.

### Key findings (measured, training-only)
- Lifetime: min 128, max 362, mean 206.31, median 199, std 46.34 (100 engines).
- Constant columns: `setting_3`, `sensor_1/5/10/16/18/19` (retained; Phase 8 ablation candidate).
- Highest-variance sensors: `sensor_9` (22.08), `sensor_14` (19.08), `sensor_4` (9.00), `sensor_3` (6.13).
- Top |corr| with RUL: `sensor_11` 0.70, `sensor_4` 0.68, `sensor_12` 0.67, `sensor_7` 0.66, `sensor_15` 0.64.
- Operating settings ~constant (std <= 0.0022; `setting_3` exactly constant) — negligible information in FD001.
- No missing/inf values, no duplicate or unordered cycles (validation passed).

### Added
- Raw C-MAPSS FD001 ingestion (`data/raw/`: `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`, `readme.txt`) downloaded from the official NASA mirror (`phm-datasets.s3.amazonaws.com/NASA/`).
- `src/rul_prediction/data/loader.py`: schema-aware loading + dataset summaries.
- `src/rul_prediction/data/validation.py`: integrity checks (column count, numeric types, missing/inf, duplicate `(engine_id, cycle)`, cycle ordering, engine cardinality, RUL length/dtype). Constant columns are reported but never removed.
- `scripts/preprocess.py` CLI with `--dataset FD001 --validate-only`.
- Tests: `tests/test_loader.py`, `tests/test_validation.py`.
- Installed `numpy`, `pandas` into `.venv`; regenerated `requirements-lock.txt`.

### Notes
- FD001 validated programmatically: 20631 train / 13096 test rows, 100 train / 100 test engines, RUL length 100.
- Seven constant columns reported (`setting_3`, `sensor_1/5/10/16/18/19`) — retained; removal deferred to Phase 8 ablation.

### Added
- Project documentation: `README.md`, `PROJECT_SPEC.md`, `CHANGELOG.md`, `LICENSE` (MIT).
- `.gitignore` excluding `.venv`, caches, artifacts, and large/generated data.
- Git repository initialized (`main` branch).
- Local `.venv` (Python 3.12; deviation from 3.11 documented in PROJECT_SPEC.md §6).
- Minimal `rul_prediction` package (`src/` layout) with version metadata.
- `pyproject.toml`, `requirements.txt`.
- Smoke tests (`tests/test_smoke.py`).