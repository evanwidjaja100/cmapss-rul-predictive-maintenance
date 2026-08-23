# M3 Final Report — Methodology M3 Scientific Repair

Date: 2026-08-17 · Repository: cmapss-rul-predictive-maintenance
Supersedes: M2 "CV-READY" declaration (git `0251dca`) — labeled
`SUPERSEDED BY METHODOLOGY M3`.

## 1. Executive summary

Methodology M3 repaired the two scientific leaks that invalidated M2's
headline numbers: calibration engines had been used as `validation_data` in
the final FD001 fit (M3-1), and outer-fold engines were inside training
control during CV (M3-2). Under the corrected nested protocol — 40/40
candidate-fold runs, a hard completeness gate, a pre-specified selection
policy, and a clean conformal recalibration on 15 held-out engines (held out
from M3 fitting and model selection; inspected during earlier project
iterations, so the interval is empirically calibrated, not pristine) — the M3
deployment model is **xgb_w90_d6** (NASA-risk champion; accuracy champion
lstm_w60_huber). Official post-hoc FD001: RMSE 26.25, NASA 60,963.8. FD004
condition-aware variant C remains the fix for the regime collapse under the
clean protocol (official post-hoc RMSE 33.66). All claims are recomputed from
saved CSVs (falsification pass) and guarded by 19 new artifact-free protocol
tests.

## 2. What was wrong (audit findings)

| ID | Severity | Finding |
|---|---|---|
| M3-1 | CRITICAL | M2 final FD001 fit passed the 15 calibration engines as `validation_data` (early stopping, checkpointing, LR schedule) |
| M3-2 | CRITICAL | M2 outer-fold engines drove early stopping AND were the fold-evaluation engines |
| M3-3 | CRITICAL | M2 CV matrix incomplete (rf_w60 fold 1 only) while claimed "8 × 5" |
| M3-4 | HIGH | "GRU w45 wins" claim contradicted by artifacts; selection not reproducible |
| M3-5 | CRITICAL | Conformal calibrated on the leaky model |
| M3-6/7 | HIGH/MED | Hardcoded training values; platform-dependent hashes |
| M3-8 | CRITICAL | FD004 variants used validation engines for early stopping + evaluation |
| M3-9..16 | HIGH/MED | Stale sensitivity, stale docs, no protocol tests, serving the leaky model, ordering/falsification, metadata, wording, config drift |

## 3. Repairs delivered

1. **Nested CV** (`src/rul_prediction/benchmark/m3.py`): per outer fold,
   68 outer-train / 17 untouched outer-eval engines; inner 58/10 splits
   (seeds 4201–4205, `random.Random(4200+fold)` on sorted IDs) control
   duration; stage-2 refits fixed-duration with NO validation data.
2. **Complete 40/40 matrix** gated by `assert_cv_complete`; summaries and
   selection blocked on partial matrices (protocol Test C).
3. **Pre-specified selection policy** (specified in the recorded development
   session before the final M3 model comparison; Git cannot prove formal
   pre-registration — see §5): PRIMARY lowest mean NASA per engine; GUARDRAIL
   pooled-SE RMSE; TIE |bias|. Applied mechanically by
   `scripts/select_m3_model.py` → `selection_decision.json` →
   `configs/final_model_m3_fd001.yaml`.
4. **Clean final fit**: 85 dev engines only, `n_estimators = median([100, 91,
   69, 97, 53]) + 1 = 92`, zero calibration contact.
5. **Clean conformal**: 15 engine scores, k = ceil((n+1)(1−α)); q(0.1) =
   66.21, q(0.2) = 44.80, q(0.3) = 41.42; formal wording limited to
   exchangeability + predefined checkpoints; arbitrary trajectories labeled
   engineering extrapolation.
6. **FD004 clean protocol**: two-stage (150/25, seed 4201) for A/B/C/D;
   variant C selected; freeze on 212 engines; 37 validation engines untouched.
7. **Configs as source of truth** (YAML-driven freeze scripts, Test E),
   **canonical hashes** (`canonical_hash.py`, Test F), **per-run metadata**
   (`fd001_outer_metadata.jsonl`), **post-hoc after calibration with
   falsification** (Test D + metric recompute).
8. **Docs live**: README/PROJECT_SPEC/CHANGELOG rewritten M3-first; stale
   terms removed; historical claims labeled.
9. **Serving**: `m1_predictor.py` + `app_m1.py` serve the M3 model with the
   recalibrated interval, per-instance fields and the extrapolation
   disclosure; no OOD language.

## 4. Final CV results (FD001, mean ± std over 5 outer folds)

| candidate | RMSE | MAE | R² | NASA total | NASA/engine |
|---|---|---|---|---|---|
| xgb_w90_d6 (**deployment**) | 28.35 ± 2.69 | 22.68 | 0.735 | 6,256.4 ± 2,726.8 | **368.02 ± 160.40** |
| lstm_w60_huber (accuracy champion) | **26.19 ± 3.44** | 21.11 | 0.767 | 7,325.8 ± 2,493.6 | 430.93 ± 146.68 |
| gru_w45_huber | 26.74 ± 3.54 | 21.26 | 0.760 | 7,496.2 ± 2,358.6 | 441.0 ± 138.7 |
| (remaining 5 candidates) | — | — | — | higher NASA or RMSE | — |

Full matrix: `experiments/m3/fd001_outer_fold_results.csv` + `_summary.csv`.
Note: under the clean protocol no deep candidate beats XGBoost on NASA, and
no candidate beats lstm_w60_huber on RMSE — M2's single-winner framing is
replaced by role-based reporting.

## 5. Final FD001 model (deployment)

- Model: XGBoost, window 90, depth 6, n_estimators 92, trained on the 85
  development engines only (`models/m3/fd001_xgb_w90_d6.joblib` +
  `fd001_scaler.joblib`).
- Config: `configs/final_model_m3_fd001.yaml` (source of truth).

## 6. Official FD001 results (post-hoc, recomputed from saved CSVs)

| metric | value |
|---|---|
| RMSE | 26.2526 |
| MAE | 21.2347 |
| R² | 0.6009 |
| NASA total | 60,963.79 |
| NASA per engine | 609.64 |
| coverage α=0.1 (q=66.21) | 99% (full-history 100%, short-history 96%) |

Predictions: `experiments/m3/fd001_official_predictions.csv`. These are
permanently post-hoc (labels inspected in the M1-0 audit); they are NOT
compared against the M2 number (23.04) as a "winner" claim — the M2
number was produced under leaky control.

## 7. Uncertainty calibration (final)

15 calibration engines, one max-|error| score per engine over five predefined
lifecycle checkpoints (0.25/0.45/0.65/0.80/0.95): `q(0.1) = 66.2097` (k=15),
`q(0.2) = 44.7955` (k=13), `q(0.3) = 41.4224` (k=12). Formal statement is
limited (exchangeability + predefined checkpoints); use on arbitrary
trajectories is an engineering extrapolation, disclosed in the app.

## 8. Error analysis

Post-hoc descriptive: official trajectories are truncated before failure —
`cycle.max()` is observed history, never lifetime. The frozen model
overpredicts (+19.6 cycles mean, 91% of engines), strongest on short observed
histories (< 90: +29.0…+49.4). Serving exposes only the objective
`history_is_padded` flag; no empirical risk threshold is applied (these
patterns are descriptive, not serving triggers).

## 9. Sensitivity analysis

Sensor occlusion on the FINAL M3 model (not SHAP): most influential —
sensors 4, 11, 3, 9, 12, 7, 20. Constant sensors (1, 5, 10, 16, 18, 19)
contribute zero (consistency check). Sensor 6 (M1's earlier flag) is nearly
inert for the M3 model. Conclusions differ from M1 by measurement, not by
forcing.

## 10. FD004 condition-aware experiment (clean protocol)

| variant | RMSE | R² | NASA/engine | pred_std |
|---|---|---|---|---|
| A | 72.41 | −0.246 | 75,343 | 0.0 (collapse) |
| B | 72.41 | −0.246 | 75,333 | 0.0 (collapse) |
| **C (selected)** | **29.83** | **0.789** | **1,449** | 70.6 |
| D | 33.97 | 0.726 | 81,963 | 75.9 |

Selection by the same pre-declared principle (NASA per engine, then RMSE):
**C**. Unlike M2, D does not beat C on RMSE under the clean protocol.
Official FD004 (post-hoc): RMSE 33.6579, MAE 22.0687, R² 0.6189, NASA
1,545,798.5, pred_std 66.84. Model: `models/m3/fd004_gru_w45_huber_condC.keras`
(config `final_model_m3_fd004.yaml`).

## 11. Verification

- Full test suite: **155 passed** (incl. 19 M3 protocol tests A–H + duration
  rules; 4 rewritten serving tests; legacy golden test passes).
- Clean public checkout (no data/, models/, experiments artifacts): **135
  passed, 11 skipped, 9 deselected** — same command as CI
  (`-m "not needs_artifacts"`), verified locally on a simulated clean tree.
- Falsification: selection re-derived from fold CSVs (Test D); official
  metrics recomputed from saved prediction CSVs; config drift asserted; CV
  numbers re-verified.
- App import smoke: `python -c "import app_m1"` passes.
- Self-audit rounds 1 & 2 (artifacts + live-doc grep): PASS (see
  `M3_REPAIR_PLAN.md`).
- Exit checklist (20 criteria): all DONE.

---

# Final Cleanup Report (M3_FINAL_CLEANUP_PLAN.md §27) — 2026-08-17

Implementing agent: opencode. Tracker: `M3_FINAL_CLEANUP_PLAN.md` (I-1..I-17,
all DONE). Commit: `ad2ab8b` (cleanup), follow-up commit for this report.

## 27.1 Cleanup Summary

What remained wrong and what changed:

- **Config truthfulness (I-6/I-7/I-8):** `configs/final_model_m3_fd001.yaml`
  carried deep-model fields (Adam, GRU 128/64, dropout 0.3, lr 0.001) while the
  deployed model is XGBoost. Regenerated as model-type-specific XGBoost config;
  freeze scripts (`run_m3_freeze.py`, `run_m3_fd004_freeze.py`) and the
  FD004 post-hoc script now resolve all deployment-critical hyperparameters
  from YAML (no hidden `max_depth`, `window`, `loss`, `epochs` constants;
  the remaining FD004 defaults — units, dropout, KMeans n_init — were
  eliminated in the repository freeze, §28).
- **Serving (I-9/I-15):** the empirical `RISK_OBSERVED_CYCLES=90`
  short-history risk flag (derived from post-hoc official labels) was removed;
  serving exposes only objective `history_is_padded` / `n_padded_timesteps`;
  uncertainty `q` moved to tracked `configs/deployment_m3_fd001.yaml`;
  broken `m3_methodology.md` report link fixed.
- **Sensitivity (I-11):** the occlusion baseline used full run-to-failure
  sensor means (future leakage) and positional groupby alignment; replaced with
  prefix-only replacement values and (engine_id, cutoff_cycle)-keyed alignment.
- **Selection (I-12):** the |bias| tie-break sorted by signed bias; fixed to
  `abs(signed_bias_mean_mean)`. Deployment candidate unchanged.
- **Conformal wording (I-13):** "first label contact" claim replaced by the
  empirical-not-pristine disclosure (engines held out from M3 fitting and
  selection, inspected in earlier iterations).
- **Provenance (I-4):** freeze metadata now records `git_commit`,
  `git_is_dirty` (historical M3 runs ran from a dirty worktree), `git_diff_hash`
  and `timestamp_utc`; the shared `run_metadata()` helper additionally supports
  `source_tree_hash` for future runs (the committed freeze metadata does not
  contain that field); historical dirty-run provenance disclosed.
- **Tracking (I-1/I-2/I-3):** `experiments/m3` committed (`.gitignore`
  exception); structural + metric-falsification tests added.
- **Docs (I-5/I-16):** "pre-registered" → "pre-specified"; real test counts;
  no stale risk-flag/OOD-adjacent claims; historical documents marked.

## 27.2 Experiment Artifact Recovery

`experiments/m3` was recovered **without rerunning the CV**: all 22 audit
files existed locally and were verified by `test_m3_experiment_dir_structurally_complete`
(40 candidate-fold rows, folds 1–5 per candidate, 15 conformal engine scores,
100 official predictions, 4 FD004 variants × 5 checkpoints, selection =
`xgb_w90_d6`, quantile alphas [0.1, 0.2, 0.3]).

```text
40/40 status: verified from fd001_outer_fold_results.csv (40 rows, 8 candidates x 5 folds)
files committed: 22 (ad2ab8b): fold results, engine-level, predictions, cv_summary,
  best_epochs, split manifest, metadata jsonl, selection_decision.json, conformal
  engine scores/quantiles/calibration json, official predictions, final metrics,
  final-fit metadata (FD001+FD004), FD004 variant results/predictions
hash/integrity checks: train/cal IDs sha256 in fd001_final_fit_metadata.json;
  structural test asserts row counts, fold coverage, selection and q alphas
```

## 27.3 Configuration Truthfulness

FD001 (`configs/final_model_m3_fd001.yaml`): `max_depth: 6`,
`learning_rate: 0.05`, `subsample: 0.8`, `colsample_bytree: 0.8`,
`n_estimators: 92`, `random_state: 42`, `early_stopping_rounds: null`,
`window: 90`, candidate `xgb_w90_d6`. No deep-model fields remain. The freeze
asserts `training_control.fixed_n_estimators == model.n_estimators` and applies
`set_params` from YAML (tests monkeypatch the model to capture them).

FD004 (`configs/final_model_m3_fd004.yaml`): GRU, `window: 45`,
`loss: huber`, variant C (KMeans k=6 per-regime scalers). `run_m3_fd004_freeze.py`
`resolve_model_config()` reads window/loss/batch_size/learning_rate/seed from
YAML and names the artifact `fd004_gru_w{window}_{loss}_cond{variant}.keras`;
post-hoc reads window/loss from YAML too. FD004 re-freeze was NOT rerun: the
existing frozen model matches the config exactly and headline metrics are
falsified from saved predictions; a re-freeze could perturb them without
evidence (allowed by §18).

## 27.4 Serving Cleanup

- **Padding behavior:** `history_is_padded = observed_cycles < window` (objective);
  `n_padded_timesteps = max(window − observed, 0)`; left-padding is the shared
  representation — never called OOD.
- **Test-derived threshold:** `RISK_OBSERVED_CYCLES` and
  `short_history_risk_flag` removed from `m1_predictor.py`/`app_m1.py`
  (grep + source test enforce absence).
- **Tracked conformal q source:** `configs/deployment_m3_fd001.yaml`
  (q 66.2097 at α=0.1, q_by_alpha 0.2→44.7955, 0.3→41.4224, 15 calibration
  engines, checkpoint fractions, `interpretation: empirical_m3_calibration`);
  `load_deployment_q()` cross-checks against the audit CSV when present.
- **Missing artifacts:** absent frozen models now raise a FileNotFoundError
  explaining how to generate them (`scripts/run_m3_freeze.py`, needs data/raw).

## 27.5 Sensitivity Correction

- **Old defect:** `engine_means = groupby("engine_id")[...].mean()` was a full
  run-to-failure mean → occlusion baseline contained future sensor values;
  delta alignment relied on positional groupby order.
- **Prefix-only replacement:** `prefix_replacement_value()` = mean over cycles
  ≤ cutoff only; `sensor_occlusion_deltas()` keyed by (engine_id, cutoff_cycle)
  — order-independent (tested with scrambled rows).
- **Corrected ranking** (`reports/tables/m3_sensor_sensitivity.csv`,
  `reports/m3_sensitivity.md`): sensor_4 16.98, sensor_11 14.18, sensor_3
  10.46, sensor_9 10.35, sensor_12 10.29, sensor_7 9.23, sensor_20 7.49, then
  21/15/2/14/17/8/13/6; constant sensors 1/5/10/16 ≈ 0 (consistency check).

## 27.6 Selection Policy

- NASA-risk champion + deployment: **xgb_w90_d6** (NASA/engine 368.02 ± 160.40).
- Accuracy champion: **lstm_w60_huber** (RMSE 26.19 ± 3.44).
- **Bias tie-break fix:** `apply_selection_policy` sorts by
  `abs(signed_bias_mean_mean)`; synthetic test (A bias −20 vs B bias +1 → B
  wins). Real deployment decision unchanged: xgb_w90_d6 vs xgb_w60_d6 NASA gap
  181 ≫ pooled SE ~118.6 → no tie engaged.

## 27.7 Conformal Interpretation

- **Mechanics:** engine-cluster split-conformal, one max-|error| score per
  held-out engine over five predefined checkpoints → 15 scores,
  k = ceil((n+1)(1−α)); q(0.1)=66.2097, q(0.2)=44.7955, q(0.3)=41.4224.
- **Historical caveat:** the 15 engines were held out from M3 fitting and
  model selection but inspected in earlier iterations → empirically calibrated
  interval, NOT a pristine one-shot external guarantee (`historical_caveat` in
  `fd001_conformal_calibration.json`; disclosure in app and README §5).
- **Formal limitations:** simultaneous coverage ≥ 1−α requires exchangeability
  under the predefined checkpoint scheme; arbitrary uploaded trajectories are an
  engineering extrapolation (labeled).
- **Post-hoc empirical coverage** (α=0.1, q=66.21): official 99%, full-history
  100%, short-history 96%.

## 27.8 Reproducibility

- **Dirty historical provenance:** historical M3 runs recorded
  `git_commit=0251dca` (pre-M3 HEAD) with no dirty flag; the M3
  implementation files were uncommitted during those runs. This is disclosed in
  the CHANGELOG cleanup entry; `fd001_outer_metadata.jsonl` remains as the
  honest historical record.
- **New metadata behavior:** re-frozen FD001 metadata carries provenance
  (`git_commit 53e50d8`, `git_is_dirty: true`, `git_diff_hash`,
  `timestamp_utc`) + `model_params` from YAML.
- **Git commits:** `ad2ab8b` (cleanup: configs, scripts, serving, tests,
  experiments/m3, docs) + follow-up (this report + .opencode hygiene).
- **Artifact tracking:** 22 files under `experiments/m3` are now public
  (`.gitignore` `!experiments/m3/`).

## 27.9 Testing

*(Cleanup-session measurements, 2026-08-17; current measurements after the
repository freeze are in §28.)*

- **Full local artifact-rich suite:** **155 passed** (16 warnings).
- **True clean-checkout suite** (temp tree without data/, models/,
  experiments/, CI command `-m "not needs_artifacts"`): **135 passed,
  11 skipped, 9 deselected, 0 failed**. *(Historical: this was a simulated
  tree; §28 replaces it with a real `git clone .` measurement.)*
- **Skips/deselections:** 11 clean skips (raw data / processed artifacts /
  legacy golden / experiments), 9 `needs_artifacts` deselections.
- **CI command:** `.github/workflows/ci.yml` runs exactly
  `python -m pytest -m "not needs_artifacts"` — matches README/PROJECT_SPEC.
- **App smoke:** `python -c "import app_m1"` passes in the artifact-rich tree;
  in a clean tree it raises the explicit generate-artifacts message (§27.4).

## 27.10 Metric Falsification

Recomputed by `tests/test_m3_cleanup.py` from saved prediction CSVs
(marked `needs_artifacts`; tolerances 1e-3 / NASA 0.05):

- FD001 (100 official predictions): RMSE **26.2526**, MAE **21.2347**,
  R² **0.6009**, NASA **60,963.79** — matches `fd001_final_metrics.json`.
- FD004 variant C (248 official predictions): RMSE **33.6579**, MAE **22.0687**,
  R² **0.6189**, NASA **1,545,798.5** — matches `fd004_final_metrics.json`.
- CV summary re-derived from fold rows (`cv_summary(fold_rows, CV_CANDIDATES)`)
  matches `experiments/m3/fd001_cv_summary.csv` for RMSE/MAE/R²/NASA_mean_per_engine/signed_bias.

## 27.11 Remaining Limitations

- Official FD001/FD004 labels are permanently post-hoc (inspected in the M1-0
  audit); no sealed-evaluation claim.
- The conformal interval is empirically calibrated, not a pristine external
  guarantee (see 27.7); arbitrary trajectories are an engineering extrapolation.
- FD004 official NASA (1.55M) ≫ validation NASA — regime transfer to the
  official test remains a limitation; the frozen model overpredicts (+19.6
  cycles mean on official FD001, 91% of engines).
- Historical M3 runs were executed from a dirty worktree (provenance
  disclosed, not reversible); the FD004 freeze metadata predates the
  provenance fields and was intentionally not re-run.
- TF/Keras training is not bit-deterministic across environments; exact metric
  reproduction relies on saved artifacts (which are now committed).
- `experiments/m3/cv_run.log` / `cv_run.err.log` are committed run logs;
  `data/raw`, `models/` and `data/processed` remain untracked by design.

## 27.12 CV Readiness

```text
CV-READY
```

Evidence: 40/40 CV matrix present and committed with folds 1–5 per candidate;
selection policy mechanically re-applied from artifacts (deployment
`xgb_w90_d6`); configs authoritative and falsified against freeze metadata;
headline metrics recomputed from saved predictions; app smoke passes; exit
checklist (40/40 items, §26) complete; working tree clean except `.opencode/`
session state. *(Test counts for this report: 155 full local / clean-checkout
135 passed, 11 skipped, 9 deselected — simulated tree; superseded by the real
`git clone .` QA in §28.)*

# 28. Repository Freeze — 2026-08-18

Implementing agent: opencode. Tracker: `M3_FINAL_FREEZE_PLAN.md`.
Master instruction (session-only, not committed). No retraining: FD004
YAML values verified equal to the historical effective runtime values
(§28.2); no FD001 CV rerun, no FD004 A/B/C/D rerun, no FD004 refit.

## 28.1 Freeze Summary

- **FD004 config authority (freeze-1):** `resolve_model_config` now resolves
  architecture, units, dropout, loss, window, learning rate, batch size, seed,
  fixed epochs, variant, n_clusters, cluster_seed and n_init — all from YAML.
  `fit_preprocessing` threads `k/seed/n_init`; `fit_condition_models` gained an
  `n_init` parameter (was hardcoded 10). `configs/final_model_m3_fd004.yaml`
  restructured: `model.units` / `model.dropout` / `model.fixed_epochs` explicit,
  `condition_preprocessing.clustering{method,n_clusters,random_state,n_init}`,
  `training` section. `write_fd004_config` emits the same structure.
- **FD001 feature metadata (freeze-2):** `configs/final_model_m3_fd001.yaml`
  now documents the real XGBoost feature path (engineered variable history via
  `extract_m1_features`, `sequence_padding_consumed_by_model: false`);
  `select_m3_model.py` parity. Tests prove the estimator receives a 2D
  engineered matrix, never `[X, mask]`.
- **Real clean-clone QA (freeze-3):** the fake simulated-checkout QA is
  replaced by a real `git clone .` of the committed repository; artifact-free
  suite run from the clone (tracked `experiments/m3` present). Counts in
  §28.3. Four tests reading tracked audit tables were unmarked from
  `needs_artifacts` so CI exercises them.
- **Wording (freeze-4/5):** "pre-registered" → "pre-specified" everywhere
  live; `source_tree_hash` claims made exact (committed metadata has
  git_commit/git_is_dirty/git_diff_hash/timestamp_utc; the helper supports
  source_tree_hash for future runs); references to the untracked master
  cleanup-plan filename replaced with the tracked `M3_FINAL_CLEANUP_PLAN.md`;
  PROJECT_SPEC calibration-label-contact wording reconciled (labels first
  touched after the final M3 fit during calibration; engines inspected in
  earlier iterations).
- **New tests (freeze-7):** config-authority (units/dropout/n_clusters/
  random_state/n_init/…), hidden-constant scan, `fit_preprocessing` threading,
  FD001 feature metadata + 2D-matrix feature path, conformal q falsification,
  provenance schema (current metadata vs helper), required tracked
  references, no broken master-plan references.

## 28.2 FD004 Old-Effective vs New YAML Values

| Parameter | Old effective runtime | New YAML-resolved | Match? |
|---|---|---|---|
| window | 45 (`WINDOW = 45`) | 45 | YES |
| units | (128, 64) (`m1_gru` default) | [128, 64] | YES |
| dropout | 0.3 (`m1_gru` default) | 0.3 | YES |
| loss | huber (`loss="huber"`) | huber | YES |
| learning_rate | 0.001 (`m1_gru` default) | 0.001 | YES |
| batch_size | 256 (`BATCH_SIZE = 256`) | 256 | YES |
| seed | 42 (`SEED = 42`) | 42 | YES |
| fixed_epochs | 8 (winner best_epoch) | 8 | YES |
| variant | C (selection) | C | YES |
| n_clusters | 6 (helper default) | 6 | YES |
| cluster_seed | 42 (helper default) | 42 | YES |
| n_init | 10 (hardcoded) | 10 | YES |

Verdict: every YAML-resolved value equals the old effective value → **no FD004
retrain required**.

## 28.3 Test Measurements (2026-08-18)

```text
Full local artifact-rich suite:
164 passed (16 warnings)

Artifact-free subset (-m "not needs_artifacts", local):
159 passed, 5 deselected (4 warnings)

Real clean Git clone (git clone . of the committed repo; PYTHONPATH=<clone>\src;
same CI command -m "not needs_artifacts"):
149 passed, 10 skipped, 5 deselected (0 failed)
```

Clone skips are the raw-data / trained-model / golden-file dependent tests that
cannot run without untracked artifacts; the tracked `experiments/m3` audit
tables ARE present in the clone, so structural, falsification, conformal and
provenance tests all executed there.