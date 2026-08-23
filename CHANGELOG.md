# CHANGELOG

## Repository integrity review wave — 2026-08-22

> Adversarial-review hardening pass on top of the 2026-08-21 implementation
> (`8529ecf`..`8ec6a00`). No retraining, no reruns; all four frozen binary
> hashes unchanged (`23bd460c…`, `d1b02cfc…`, `9b39a059…`, `f22ef718…`).
> Two independent read-only reviews (repository/CI/docs and
> provenance/manifest/FD004) reported 0 P0, 8 P1, ~16 P2 findings; all P1/P2
> items are fixed or explicitly documented below.

- **Provenance hash semantics (P1):** `source_tree_hash` now hashes current
  worktree bytes for git-tracked execution inputs — staged/unstaged edits
  change the digest (Python executes worktree bytes, not HEAD). Untracked
  directories no longer crash collection; hashing failures are fail-closed.
- **Dirty-run policy wired in (P1):** FD004 variant-run, freeze, and post-hoc
  entrypoints call `assert_reproducible_run_state()` before any training,
  loading, or output write; dirty execution requires `--allow-dirty-reason`
  plus `--allow-dirty-snapshot-dir`. Snapshot destinations inside `src/` or
  `scripts/` are rejected outright.
- **FD004 freeze fail-closed gates (P1):** lenient config fallback deleted;
  immutable-baseline gate refuses overwriting a condition joblib whose bytes
  match historical SHA `f22ef718…`; canonical-config guard compares resolved
  absolute paths so absolute-path invocations cannot bypass the A/B/C/D
  completeness requirement.
- **Split evidence portability (P1):** config and scripts now reference the
  exact tracked case `experiments/splits/FD004_m1_seed42.json`; split loading
  validates exact-case names (Linux clean clones) and separately named RAW
  file hashes (`split_provenance_file_sha256`,
  `validation_cutoff_manifest_file_sha256`) pin exact split/cutoff bytes
  alongside the canonical engine-ID digests. Numerical values unchanged.
- **Contracts survive `python -O` (P1):** deployment-config asserts in
  `serving/m1_predictor.py`, leakage/completeness gates in `benchmark/m3.py`
  replaced with explicit exceptions; shared `benchmark.m1.make_predictor`
  requires keyword-only positive-int `window` (14 call sites updated).
- **Manifest verification classes (P2):** tampered/structurally invalid
  manifests are a hard integrity failure; absent manifests load with an
  explicit UNVERIFIED legacy warning (never silent); `--check` never writes
  (including the FD004 canonical predictions CSV); builder/verifier
  dirty-tree asymmetry documented.
- **Test infrastructure honesty:** manifest determinism/preservation tests no
  longer mutate the live repository (builder root-override test uses a
  disposable clone); CRLF-normalized comparisons so `autocrlf` checkouts do
  not produce false drift; headless Streamlit smoke scans output for
  tracebacks instead of relying on an alive-only timing criterion.

## M3 final repository freeze — 2026-08-18

> Audit/freeze pass (`M3_FINAL_FREEZE_PLAN.md`; master instruction is
> session-only and not committed). No retraining: FD004 YAML values verified
> equal to the historical effective values (window 45, units (128,64),
> dropout 0.3, loss huber, lr 1e-3, batch 256, seed 42, fixed_epochs 8,
> variant C, k=6, cluster_seed 42, n_init 10). No FD001 CV rerun, no FD004
> A/B/C/D rerun, no FD004 refit.

- **FD004 now fully YAML-driven (freeze-1):** the freeze path previously
  consumed units/dropout/KMeans n_init via function defaults and hardcoded
  `n_init=10` / `k=6` / `seed=42` in `fit_condition_models`. Now
  `resolve_model_config` returns architecture/units/dropout/loss/lr/batch/
  seed/fixed_epochs/variant/n_clusters/cluster_seed/n_init, `fit_preprocessing`
  threads k/seed/n_init, and `configs/final_model_m3_fd004.yaml` is
  restructured (explicit `model.units`, `model.dropout`, `model.fixed_epochs`,
  `condition_preprocessing.clustering{method,n_clusters,random_state,n_init}`
  with numeric values) — every deployment-critical hyperparameter is
  config-resolved, no hidden defaults.
- **FD001 feature metadata corrected (freeze-2):** `final_model_m3_fd001.yaml`
  now describes the real XGBoost path (engineered variable history via
  `rul_prediction.features.m1_features.extract_m1_features`,
  `sequence_padding_consumed_by_model: false`); `select_m3_model.py`
  emits the same structured block.
- **Real clean-clone QA (freeze-3):** the previous "clean public checkout"
  measurement was a simulated tree (robocopy). The QA is now a real
  `git clone .` of the committed repository; artifact-free
  (`-m "not needs_artifacts"`) suite measured from the clone. Snapshot
  at commit `23cc934` on 2026-08-18 (historical, not a permanent current count):
  full local **164 passed (16 warnings)**; artifact-free local **159 passed,
  5 deselected (4 warnings)**; real clean clone **149 passed, 10 skipped,
  5 deselected** (0 failed; skips = raw-data/model-dependent tests absent from
  a clone). Later commits change collection/counts; CI is authoritative for
  current branch health.
- **Provenance wording (freeze-4):** the committed `fd001_final_fit_metadata.json`
  records `git_commit`, `git_is_dirty`, `git_diff_hash`, `timestamp_utc`; it
  does NOT contain `source_tree_hash` (the `run_metadata()` helper supports
  that field for future runs). Docs now say exactly this.
- **Pre-registered → pre-specified (freeze-5):** live docs now say
  "pre-specified" (the selection rule was specified in the recorded
  development session before the final comparison; Git cannot prove formal
  pre-registration). Superseded/historical entries keep old wording with
  markers.
- **Broken references fixed (freeze-6):** the untracked master cleanup-plan
  filename was referenced by 4 files; all references replaced with the tracked
  `M3_FINAL_CLEANUP_PLAN.md`.
- **Tests (freeze-7):** added config-authority, feature-path, provenance,
  conformal-q falsification, metric-falsification (tracked tables), and
  broken-reference tests; 4 cleanup tests reading tracked `experiments/m3`
  tables unmarked from `needs_artifacts` so CI exercises them.

## M3 final cleanup — 2026-08-17

> Final reproducibility / auditability pass
> (`M3_FINAL_CLEANUP_PLAN.md`). No full CV rerun; all headline numbers were
> recomputed from saved artifacts and reproduced unchanged (FD001 post-hoc
> RMSE 26.2526 / NASA 60,963.79; FD004 post-hoc RMSE 33.6579; deployment
> `xgb_w90_d6`).

- **Selection wording corrected:** "pre-registered" claims in earlier entries
  were replaced by "pre-specified" (the rule is specified in the recorded
  development session; Git cannot prove formal pre-registration).
- **Config truthfulness:** `configs/final_model_m3_fd001.yaml` regenerated
  as a model-type-specific XGBoost config (max_depth 6, learning_rate 0.05,
  subsample 0.8, colsample_bytree 0.8, n_estimators 92, random_state 42);
  FD001/FD004 freeze and post-hoc scripts now consume the YAML (no hidden
  window/loss/epochs constants).
- **Serving cleaned:** empirical `RISK_OBSERVED_CYCLES` / short-history risk
  flag removed; uncertainty `q` read from tracked
  `configs/deployment_m3_fd001.yaml`; only objective `history_is_padded`
  remains; broken `m3_methodology.md` report link fixed.
- **Sensitivity corrected:** prefix-only occlusion baseline (no future sensor
  leakage), deterministic (engine_id, cutoff_cycle) alignment; corrected
  ranking: sensors 4, 11, 3, 9, 12, 7, 20 (earlier order 4/11/12/9/3/20/7
  was based on the leaking baseline and is superseded).
- **Conformal wording:** calibration engines were held out from M3 fitting
  and model selection but inspected in earlier iterations → the interval is
  empirically calibrated, not a pristine one-shot external guarantee
  (`historical_caveat` in `fd001_conformal_calibration.json`).
- **Provenance:** freeze metadata now records `git_commit`, `git_is_dirty`
  (historical M3 runs were executed from a dirty worktree), `git_diff_hash`
  and `timestamp_utc`; the shared `run_metadata()` helper additionally
  supports `source_tree_hash` for future experiment runs (the committed
  freeze metadata does not contain it — see freeze-4 above).
- **Metric falsification tests:** saved official predictions recompute the
  headline FD001/FD004 metrics; CV summaries re-derived from fold rows;
  selection policy (incl. |bias| tie-break) re-applied.
- **Test counts (real measurements):** full local artifact-rich suite
  **155 passed**; clean public checkout (no data/, models/, experiments
  artifacts, CI command `-m "not needs_artifacts"`) **135 passed, 11 skipped,
  9 deselected**. Earlier "134 / 130" counts in the M3 entry below are
  superseded.
- **Tracking:** `experiments/m3` audit artifacts (CV matrix, selection,
  conformal, post-hoc) are committed (`.gitignore` exception added).

## Methodology M3 (repair) — 2026-08-17

> M3 repairs the M2 audit findings (`M3_REPAIR_PLAN.md`, M3-1 …
> M3-16). M1 and M2 remain historical record, labeled
> `SUPERSEDED BY METHODOLOGY M3` where they conflict with current numbers.

- **Calibration leakage in final FD001 fit fixed (M3-1):** the M2 freeze
  passed the 15 calibration engines as `validation_data` to `model.fit()`
  (training control = weights/epochs touched by calibration labels). M3
  final fit runs on the 85 development engines only, fixed duration
  (XGBoost `n_estimators = round(median(best_iteration)) + 1 = 92`), zero
  calibration contact (`run_m3_freeze.py`; `final_fit_metadata.json`).
- **Outer-fold isolation fixed (M3-2):** M2 ran outer-fold early stopping
  with the 17 outer-evaluation engines inside the training loop. M3 nested
  CV: per fold 68 outer-train / 17 untouched outer-eval; inner 58/10 splits
  (seeds 4201–4205, `random.Random(4200+fold)` on sorted IDs, documented
  pre-run) control duration only; stage-2 retrains fixed-duration with NO
  validation data. `src/rul_prediction/benchmark/m3.py`,
  `experiments/m3/fd001_outer_fold_results.csv`.
- **Complete CV matrix (M3-3):** M2's claimed 8×5 matrix had only 40/40
  requested but rf_w60 delivered fold 1 only (39/40) and was presented
  complete. M3 runs all 8 candidates × 5 folds = **40/40**; summaries are
  gated by `assert_cv_complete` (exact row count + fold coverage + manifest
  hashes) and a protocol test rejects partial matrices.
- **Model-selection claims fixed (M3-4):** M2 named one winner
  (gru_w45_huber) that was neither lowest-RMSE nor lowest-NASA. M3
  pre-specified the rule (PRIMARY lowest mean NASA/engine; GUARDRAIL
  pooled-SE RMSE; TIE |bias|) and reports roles separately: accuracy champion
  `lstm_w60_huber` (RMSE 26.19±3.44), NASA-risk champion + deployment
  **`xgb_w90_d6`** (NASA/engine 368.02±160.40, RMSE 28.35±2.69).
  `selection_decision.json`, `configs/final_model_m3_fd001.yaml`.
- **Conformal rebuilt (M3-5):** recalibrated on the clean final model with
  15 held-out calibration engines (one max-|error| score per engine over the
  predefined checkpoints); q(0.1)=66.21, q(0.2)=44.80, q(0.3)=41.42; formal
  coverage wording limited (exchangeability + predefined checkpoints);
  arbitrary-trajectory use labeled engineering extrapolation; post-hoc
  official coverage 99% at α=0.1.
- **Configs drive training (M3-6):** freeze scripts consume
  `configs/final_model_m3_fd001.yaml` / `_fd004.yaml`; protocol test E
  rejects any seed not derivable from config; F verifies canonical manifest
  hashes. `canonical_hash.py` (platform-independent canonical JSON/CSV).
- **FD004 clean protocol (M3-7):** M2's variant comparison used
  validation engines for early stopping (calibration contact). M3 runs
  A/B/C/D under the two-stage protocol (150/25 inner split, seed 4201, eval
  on 37 untouched engines): A/B collapse (pred_std 0.0), C RMSE 29.83 /
  R² 0.789 / NASA 1,449 per engine, D RMSE 33.97 / NASA 81,963 per engine —
  C selected (per NASA, then RMSE); D's M2 RMSE edge is gone under clean
  control. Freeze on 212 (175+37); official post-hoc FD004: RMSE 33.66,
  R² 0.619, NASA 1,545,798.5.
- **Sensitivity rerun (M3-8):** `explain_m3_sensitivity.py` occludes on
  the frozen M3 model; sensors 4/11/3/9/12/7/20 most influential; constant
  sensors (1/5/10/16/18/19) contribute zero; sensor 6 (M1's flag) nearly
  inert now — reported as measured, not forced. `reports/m3_sensitivity.md`.
- **Docs live (M3-9):** README/PROJECT_SPEC/CHANGELOG updated to M3-first
  with stale M2 claims (winner, q=70.34, coverage 98%, "8 candidates x 5
  folds", test counts) annotated; `app_m1.py` labels
  "SUPERSEDED BY METHODOLOGY M3".
- **Protocol tests (M3-10):** `tests/test_m3_protocol.py` (artifact-free,
  19 tests) — Tests A–H + duration rules.
- **Serving (M3-11):** `m1_predictor.py` reads the M3 config YAML +
  recalibrated quantiles; app shows model version / observed cycles /
  `history_is_padded`+count / interval / calibration method / disclosure.
- Tests: 115 → **134 passing** (19 protocol + 4 rewritten serving + others);
  artifact-free CI subset re-verified. *(counts superseded by the M3 final
  cleanup entry: 154 full local / clean checkout 134 passed, 11 skipped,
  9 deselected.)*
- Artifacts under `experiments/m3/`, `models/m3/`, `reports/m3_*.md`,
  `reports/tables/m3_*`; commit `m3` tag target.

## Methodology M2 (repair) — 2026-08-15 (`67a0e58` + follow-ups)

> M2 repairs four audit findings (`M2_REPAIR_PLAN.md`, R1–R20). M1
> conclusions that are superseded are labeled `SUPERSEDED BY METHODOLOGY M2`
> in their reports; M1 remains historical record.

- **Lifetime semantics corrected (R1/R2/R3/R4):** official test trajectories
  are truncated before failure, so `cycle.max()` is observed history, never
  lifetime. `implied_failure_cycle = observed_cycles + true_rul` is the
  labeled quantity. The M1 claim "44 engines with lifetime < 128 carry 99.8%
  of NASA" has **no lifetime-based counterpart — that group is empty on the
  official test**; it was an observed-history artifact. Serving no longer
  classifies OOD: `history_is_padded` (objective) + empirical short-history
  risk flag. `reports/m2_error_analysis.md`.
- **Engine-group 5-fold CV selection (R5/R6/R7/R8):** 85 development / 15
  calibration engines (M1 calibration IDs preserved), 5-fold group CV
  (seed 42), balanced fractions 0.25/0.45/0.65/0.80/0.95 fixed pre-comparison,
  8 bounded candidates. Winner **gru_w45_huber** (RMSE 24.45±3.62, NASA
  5,828±4,451, bias −0.25). Frozen on 85 dev engines: official post-hoc
  RMSE 23.04 / R² 0.693 / NASA 6,700.59 (11.5× vs M1's 77,387.53).
  `configs/final_model_m2_fd001.yaml`, `reports/m2_cross_validation.md`.
- **Engine-cluster conformal (R9/R10/R11):** one max-|error| score per
  calibration engine over its 5 checkpoints (n=15), k = ceil((n+1)(1−α))
  clamped; α=0.1 → q = 70.34; official coverage 98%. M1-8's n=75 row-level
  q=24.10 superseded. `reports/m2_conformal.md`.
- **FD004 condition-aware modeling (R13/R14):** A/B (global scaler ± settings)
  collapse reproduced (pred std 0.0, RMSE ~72); **C (KMeans k=6 per-regime
  scalers) restores variance** — RMSE 29.37, R² 0.795, NASA 46,165 (55× vs A);
  D (C + one-hot) RMSE-optimal (27.22) but NASA 68,981 — not selected. Frozen
  C official post-hoc: RMSE 33.83 / NASA 1,345,518 (1.9× better than the
  M1-11 collapse). FD004 official = post-hoc by policy from M2 on.
  `configs/final_model_m2_fd004.yaml`, `reports/m2_fd004.md`.
- **Engineering (R15–R20):** `explain_m1_shap.py` renamed
  `explain_m1_sensitivity.py` (occlusion, not SHAP); `pyproject.toml`
  `requires-python = ">=3.11,<3.13"`; `THIRD_PARTY_NOTICES.md`; CI markers
  declared; `experiments/splits/` + `experiments/m2/` un-ignored for
  auditability; M1 reports bannered SUPERSEDED; README/PROJECT_SPEC updated.
- Tests: 110 → **115 passing** (16 M2 methodology + 5 FD004 condition +
  rewritten serving tests); app smoke `python -c "import app_m1"` passes.

## Methodology M1 (Phases M1-0 … M1-12) — 2026-08-15

> **Correction (M1-12):** legacy entries below claim the official test set was
> "evaluated exactly once". The official FD001 labels were re-inspected during
> the M1 audits, so from M1 onward all official FD001 results are labeled
> **post-hoc** (see `AUDIT_M1.md` Issues 1 & 7).

### M1-0 — Baseline audit (`c3c8694`)

- `AUDIT_M1.md`: 11 confirmed issues (capped-target headline, "exactly once" claims, missing attribution, absolute lock path, skipped roadmap phases); legacy Phase 1–10 experiment labeled (`configs/legacy_cap45_model.yaml`, `reports/legacy/README.md`); no code or results modified.

### M1-1 — Methodology setup (`2aedc42`)

- M1 plan: primary target = **raw RUL** (no cap); 70/15/15 engine split (seed 42); fixed pseudo-test manifests (75 validation + 75 calibration rows, 5 lifecycle fractions); shared runner `src/rul_prediction/benchmark/m1.py`; consistent padded+masked history for train and inference.

### M1-2 — M1 preprocessing (`2c93185`)

- `src/rul_prediction/data/m1_preprocessing.py` + `scripts/preprocess_m1.py`: raw-RUL targets, scaler fit on train engines only, artifacts under `data/processed/m1/FD001/raw/`.

### M1-3 — Raw-RUL benchmark (`e88388b`)

- mean/linear/rf/xgboost/lstm/gru/tcn on the fixed 75-row validation manifest; recurrent models dominate (gru w30: RMSE 24.66); linear regression unusable on raw RUL.

### M1-4 — Window & hyperparameter ablations (`67664ce`)

- 45 runs (windows 15–90, loss, dropout, depth); **selection: GRU w45 huber — validation RMSE 13.74, MAE 9.69, R² 0.877, NASA 200.01**. `reports/m1_ablation.md`, `reports/tables/m1_ablation_results.csv`.

### M1-5 — Freeze + post-hoc FD001 (`be9252a`)

- `models/m1_frozen_gru_w45_huber.keras`; validation reproduced bit-exact (13.7406); official FD001 post-hoc: RMSE 29.0377, MAE 19.1715, R² 0.5117, NASA 77,387.53; 5 late misses = 96.6% of NASA. `reports/m1_freeze.md`.

### M1-6 — Error analysis (`b3742bc`)

- Training lifetime min 128; official engines below it (44/100) are missed late 80% of the time and carry 99.8% of NASA; in-range engines: mean error −4.3 cycles. `reports/m1_error_analysis.md`.

### M1-7 — Explainability (`74b4ce2`)

- shap 0.52.0 (keras 3.15.1 breaks Deep/Gradient/Kernel explainers — documented); exact leave-one-sensor-out attribution; sensors 2/4/6/7/8 flip sign (late-miss overprediction); constant sensors ≈ 0 attribution; non-additivity disclosed. `reports/m1_explainability.md`.

### M1-8 — Conformal uncertainty (`3d3f644`)

- Split-conformal calibration on the 75 calibration rows; 90% interval width 24.10 cycles; coverage: calibration 92%, validation 88%, official 69% (85.7% in-range vs 47.7% OOD); lower-bound predictor cuts official NASA 3.2× at α=0.2. `reports/m1_conformal.md`.

### M1-9 — Streamlit serving (`815e718`)

- `app_m1.py` + `src/rul_prediction/serving/m1_predictor.py` (predictions bit-identical to the freeze, golden-tested); 90% conformal intervals, alarm lower bound, OOD flag; `reports/m1_serving.md`.

### M1-10 — CI / dependency cleanup (`978b060`)

- `requirements-lock.txt` regenerated (path-free editable line; includes shap/streamlit); Python version policy reconciled (>= 3.11, tested on 3.12); `needs_artifacts` marker registered; GitHub Actions workflow runs the artifact-free subset (78 passed, verified on an artifact-free tree).

### M1-11 — FD004 generalization study (`27a3e64`)

- FD004 acquired (Kaggle mirror; NASA S3 403 for all probed paths); **sealed-labels gate**: repo-wide grep proved no code reads `RUL_FD004.txt`, sha256 pinned at download and re-verified before the first-ever label read. The exact GRU recipe **does not transfer**: collapses to a constant (official RMSE 64.42, R² −0.40, NASA 2,663,846.31; validation R² −2.23) under 6 operating conditions. Condition-aware preprocessing documented as future work. `reports/m1_fd004.md`.

### M1-12 — Final documentation & attribution

- README rewritten: raw-RUL headline with post-hoc labeling, M1 pipeline/reproduction, FD004 verdict, attribution (NASA C-MAPSS + `aun151214/predictive-maintenance-cmapss`).
- "Exactly once" claims corrected (labeled historical/post-hoc) in `reports/phase9_final_evaluation.md`, `reports/phase10_serving.md`, `configs/final_model.yaml`, `scripts/final_evaluation.py`.
- Mojibake check on legacy reports: verified clean (valid UTF-8); `reports/phase8_ablation.md` "6918 engines" typo → "sequences/windows"; notebook absolute path made machine-neutral.
- PROJECT_SPEC.md: M1 status section mapping roadmap phases 10–16.

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
