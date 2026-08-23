﻿# M3_REPAIR_PLAN.md

> **SUPERSEDED NOTE (2026-08-17, M3 final cleanup):** "pre-registered" in
> this historical document means the policy was specified in the recorded
> development session before M3 results were inspected. Git cannot prove a
> formal pre-registration, so the cleanup pass
> (`M3_FINAL_CLEANUP_PLAN.md`, CHANGELOG "M3 final cleanup") uses
> "pre-specified". Test counts stated here (134/130) are superseded by the
> cleanup measurements (155 full local / clean checkout 135 passed,
> 11 skipped, 9 deselected).

Methodology M3 â€” final scientific repair of the cmapss-rul-predictive-maintenance repository.
Supersedes the M2 "CV-READY" declaration (git 0251dca). No V1/M1/M2 history is deleted;
superseded conclusions are labeled `SUPERSEDED BY METHODOLOGY M3`.

Pre-registered selection policy (locked BEFORE any M3 CV results are inspected):

```
FD001 deployment selection rule (pre-declared, M3):
  PRIMARY  : lowest mean NASA per engine (macro over 5 outer folds)
  GUARDRAIL: if the two top candidates' NASA means differ by less than one
             pooled standard error  SE = sqrt((s1^2 + s2^2) / n_folds),
             prefer lower mean RMSE.
  TIE      : prefer smaller absolute signed bias.
  Report separately: accuracy champion (lowest RMSE), NASA-risk champion
  (rule winner), deployment selection. They may differ; never call all "best".

FD004 variant selection rule (same principle, documented at config time):
  PRIMARY  : lowest NASA per engine on the 37 held-out validation engines
  SECONDARY: lower RMSE   TERTIARY: smaller |signed bias|
```

Inner early-stop split (FD001): within each outer fold's 68 outer-training engines,
`random.Random(4200 + fold)` shuffles the sorted engine IDs; first 58 = inner-fit,
last 10 = inner-stop (58/10 â‰ˆ 85/15). Seeds: fold1=4201 â€¦ fold5=4205.
FD004: within 175 training engines, same scheme gives 150 inner-fit / 25 inner-stop.

Final duration rules (derived ONLY from development-only training control):
```
deep models : final_epoch_count = round(median(best_epoch_per_outer_fold))   (best_epoch = argmin(val_loss)+1, inner-stop)
XGBoost     : final_n_estimators = round(median(best_iteration_per_outer_fold)) + 1
RandomForest: no early stopping; final fit on all 85 dev engines
```

| ID | Issue | Severity | Scientific consequence | Files/code affected | Planned correction | Required tests | Required experiment | Status | Evidence |
|----|-------|----------|------------------------|---------------------|--------------------|-----------------|----------------------|--------|----------|
| M3-1 | Final FD001 fit uses the 15 calibration engines as `validation_data` (EarlyStopping, checkpoint, LR schedule) | CRITICAL | Calibration labels influence weights/epochs/checkpoints; split-conformal guarantee is void; official numbers inflated | scripts/run_m2_freeze.py:84-92, src/rul_prediction/benchmark/m1.py:train_model | New run_m3_freeze.py: fit scaler+model on 85 dev engines ONLY, fixed epoch count from development-only CV, NO validation_data | Test A: calibration isolation (final train IDs disjoint from cal IDs; no validation_data path) | Freeze M3 final model | DONE | run_m3_freeze.py fits on 85 dev engine IDs only; Test A passes; models/m3/fd001_xgb_w90_d6.joblib + fd001_scaler.joblib; metadata records engine hashes; no validation_data path exists |
| M3-2 | Outer-fold validation engines drive early stopping AND are then used for reported fold metrics | CRITICAL | Outer-fold RMSE/NASA are optimistic (in-sample control selection); selection biased | src/rul_prediction/benchmark/m2.py:run_cv_fold (partition_sequences on val_ids â†’ train_model), run_m2_cv.py | Nested design: Stage 1 (58 inner-fit / 10 inner-stop, seed 4200+fold) finds best epoch/iteration; Stage 2 refits preprocessing on all 68 outer-train and retrains fixed-duration; evaluate 17 untouched outer-eval engines | Test B: outer-eval IDs absent from inner fit/stop/preprocessing fit/training control | Full 40-run nested CV | DONE | benchmark/m3.py implements the two-stage protocol; inner seeds 4201-4205 recorded per fold; Test B passes; experiments/m3/fd001_outer_fold_results.csv (40/40) |
| M3-3 | CV matrix incomplete: rf_w60 has only fold 1 (folds 2-5 missing) despite "8 candidates x 5 folds" claim | CRITICAL | Summary mixes 5-fold and 1-fold candidates; selection claims invalid | experiments/m2/fd001_cv_fold_results.csv, run_m2_cv.py (no completeness gate) | Rebuild complete 8x5=40 matrix under corrected nested protocol in experiments/m3/; hard completeness assertion blocks summary unless every candidate has folds {1..5} and 40 rows | Test C: synthetic table missing fold â†’ summary/freeze fails loudly | 40/40 candidate-fold runs | DONE | assert_cv_complete gated the summary; matrix is 8 candidates x 5 folds = 40 rows; Test C passes |
| M3-4 | Documentation claims "GRU w45 wins RMSE/MAE/R2" yet artifacts contradict; selection was post-hoc and non-reproducible | HIGH | Model-selection claim not reproducible; metric cherry-picking risk | configs/final_model_m2_fd001.yaml selection block, README, reports/m2_cross_validation.md | Pre-register selection policy (above) in this plan before any M3 CV; scripts/select_m3_model.py applies it mechanically and writes selection_decision.json + final YAML | Test D: policy applied to summary reproduces selected candidate | Selection run after CV | DONE | Policy locked in this file pre-run; selection_decision.json records NASA champion xgb_w90_d6 (368.02/engine), accuracy champion lstm_w60_huber (RMSE 26.19); Test D passes |
| M3-5 | Conformal built on the leaky M2 final model | CRITICAL | Calibration contact in training voids the exchangeability guarantee | scripts/calibrate_m2_conformal.py | run_m3_conformal.py after clean final fit: 15 engine scores (max |err| over 5 cutoffs), k=ceil((n+1)(1-alpha)) clamped, q's, coverage, limited formal wording | Test G: exactly 15 scores; cal IDs absent from training/control manifests | Recalibration on clean final model | DONE | run_m3_conformal.py: 15 engine scores; q(0.1)=66.2097 (k=15), q(0.2)=44.7955 (k=13), q(0.3)=41.4224 (k=12); post-hoc official coverage 99% (alpha 0.1); Test G passes; formal wording limited, extrapolation labeled |
| M3-6 | Freeze scripts hardcode values (WINDOW=45, k=6, epochs, variant C) instead of consuming YAML; YAML not source of truth | HIGH | Tracked config can drift from actual training | scripts/run_m2_freeze.py, scripts/run_m2_fd004_freeze.py, configs/* | configs/final_model_m3_fd001.yaml + final_model_m3_fd004.yaml; freeze scripts derive EVERY configurable value from YAML; no duplicate constants | Test E: config-driven training (temp YAML value reaches the parser) | Freeze runs | DONE | run_m3_freeze.py reads final_model_m3_fd001.yaml (window 90, n_estimators 92); run_m3_fd004_freeze.py reads final_model_m3_fd004.yaml (best_epoch 8, condition C); Test E passes |
| M3-7 | Manifest hashes are platform-dependent (raw JSON/text bytes; LF vs CRLF, whitespace, key order) | MEDIUM | Integrity verification is not reproducible across platforms | configs (sha256 fields), run_m2_freeze.py:65-67 | src/rul_prediction/data/canonical_hash.py: canonical JSON (sort_keys, separators, ensure_ascii=False, utf-8) and canonical CSV (sorted frame â†’ '\n' serialization); hashes recomputed for M3 manifests | Test F: semantically identical manifests with different formatting/line endings/key order â†’ same hash | Hash verification in freeze/conformal scripts | DONE | canonical_hash.py + Test F pass; M3 manifests hash-verified; build_m3_manifests.py asserted equality with M2 manifests |
| M3-8 | FD004 A/B/C/D use the 37 validation engines for early stopping AND variant evaluation; condition preprocessing fit on train+val-influenced control | CRITICAL | FD004 variant comparison optimistic; condition models touch validation rows indirectly through training control | scripts/run_m2_fd004.py:104-117 | Two-stage protocol: inner-fit (150) / inner-stop (25) within 175 training engines; variant best epoch from inner-stop only; rebuild preprocessing on all 175; evaluate 37 untouched validation engines; fit IDs asserted | Test H: KMeans/cluster-scaler fit IDs âŠ† allowed training IDs; val/official test IDs absent from fitting | FD004 A/B/C/D rerun | DONE | run_m3_fd004.py ran A/B/C/D under the protocol; C selected (RMSE 29.83, NASA/engine 1448.64); Test H passes; freeze on 212 engines, 37 reserved untouched |
| M3-9 | Sensitivity analysis still targets the old M1 model/artifacts and uses SHAP filenames | MEDIUM | Explainability does not describe the deployed model | scripts/explain_m1_sensitivity.py, experiments/m1_shap_*.csv | scripts/explain_m3_sensitivity.py: sensor occlusion / counterfactual attribution on the FINAL M3 model; outputs reports/m3_sensitivity.md + reports/tables/m3_sensor_sensitivity.csv + m3_temporal_sensitivity.csv; terminology "sensitivity/occlusion/attribution", never "SHAP" | artifact-free unit: occlusion function on tiny synthetic model | Rerun on M3 model | DONE | explain_m3_sensitivity.py ran on models/m3/fd001_xgb_w90_d6.joblib; outputs written; no SHAP language; occlusion unit test in test_m3_protocol.py passes |
| M3-10 | README/PROJECT_SPEC/CHANGELOG live claims are stale (test counts 93/78, M1-first ordering, "8 candidates x 5 folds", old winner claims) | HIGH | Live docs contradict artifacts | README.md, PROJECT_SPEC.md, CHANGELOG.md | README leads with Methodology M3; historical sections demoted; PROJECT_SPEC current section describes M3; CHANGELOG entry with SUPERSEDED banners; stale-term grep sweep | â€” | Documentation falsification grep | DONE | README rewritten M3-first (17 sections, legacy demoted); PROJECT_SPEC Â§3/Â§10 = M3; CHANGELOG M3 entry added; stale-term grep verified clean |
| M3-11 | No tests catch protocol violations (calibration contact, outer-eval contact, incomplete matrix, policy drift, config drift, hash fragility, conformal count, FD004 fit-IDs) | HIGH | Protocol regressions pass CI silently | tests/, .github/workflows/ci.yml | tests/test_m3_protocol.py with Tests A-H (artifact-free, synthetic); CI runs them | Tests A-H as above | â€” | DONE | tests/test_m3_protocol.py: 19 tests (A-H + duration rules) all pass; CI already runs the artifact-free subset incl. these |
| M3-12 | Streamlit serves the leaky M2 model, missing M3 disclosures and per-instance fields | MEDIUM | Live demo presents superseded model + overstated uncertainty | app_m1.py, src/rul_prediction/serving/m1_predictor.py | Serve M3 final model; show model version, raw RUL, observed cycles, history_is_padded + padded count, conformal interval, calibration method; exact engineering-extrapolation disclosure; no OOD language | app import smoke (artifact-free import test) | Streamlit smoke run | DONE | m1_predictor.py serves xgb_w90_d6 via YAML + recalibrated q; app_m1.py shows version/padded/interval/disclosure; `python -c "import app_m1"` passes |
| M3-13 | Post-hoc official evaluation must run strictly after CVâ†’selectionâ†’freezeâ†’calibration; M3 headline metrics must be recomputed from saved prediction CSVs (falsification) | HIGH | Ordering violation / trusting stored summaries | scripts/run_m3_posthoc.py, falsification pass | Order enforced by scripts; independent recompute of RMSE/MAE/R2/NASA/coverage from experiments/m3/*_predictions.csv; CV summaries recomputed from fold CSV; selection programmatically verified | Test D reuse + metric falsification helper test | Post-hoc runs + falsification | DONE | run_m3_posthoc.py recomputed official FD001 (RMSE 26.2526, MAE 21.2347, R2 0.6009, NASA 60963.79) from saved CSVs; config-drift + prediction-recompute falsification passed; coverage recomputed (99%) |
| M3-14 | Experiment metadata incomplete (no git commit, engine IDs, seeds, versions per run) | MEDIUM | Reproducibility claims unverifiable | all run scripts | run_metadata() helper: git commit, dataset, candidate, fold, inner seed, engine IDs/hash, window/features/preprocessing/model hyperparams, best epoch/iteration, final epoch count, training time, RMSE/MAE/R2/NASA/signed bias, software versions | unit: metadata helper smoke | written for every run | DONE | experiments/m3/fd001_outer_metadata.jsonl (per-run metadata with commit, seeds, engine hashes, versions); metadata helper unit test passes |
| M3-15 | FD001 m2_vs_m3 duplicate-calibration: conformal must be recalibrated with the clean final model; coverage reported as post-hoc empirical, formal wording limited to exchangeability + fixed checkpoints | HIGH | Overstated guarantee wording | reports/m3_conformal.md, run_m3_conformal.py | Limited formal statement + "engineering extrapolation" for arbitrary trajectories | Test G | Conformal rerun | DONE | reports/tables/m3_conformal_coverage.csv + conformal_calibration.json; wording limited (exchangeability + predefined checkpoints); app shows the disclosure |
| M3-16 | CV summary / fold results and YAML CV-metric consistency is fragile (config vs summary drift) | MEDIUM | Frozen config can silently diverge from measured CV | configs/final_model_m3_fd001.yaml + select script | selection_decision.json holds the verified CV numbers; freeze asserts config CV numbers == summary numbers; Test D re-derives selection from fold CSV | Test D | â€” | DONE | selection_decision.json holds verified CV numbers; final YAML embeds them; Test D re-derives selection from fold CSV and matches |

Exit checklist tracked in the FINAL section of this file (updated only when verified).

## Superseded statements (labeled, not deleted)
- M2 "CV-READY" (git 0251dca) â€” SUPERSEDED BY METHODOLOGY M3: outer folds were not held out (M3-2) and the 8x5 matrix was incomplete (M3-3).
- M2 final FD001 freeze / conformal numbers â€” SUPERSEDED BY METHODOLOGY M3: calibration contact (M3-1) and leaky control (M3-2, M3-8).
- M2 selection rationale (configs/final_model_m2_fd001.yaml) â€” SUPERSEDED: selection policy is now pre-registered (above) and mechanically applied.

## Self-audit round 1 â€” artifact + code audit (2026-08-17)

| Check | Result |
|---|---|
| Final fit engine IDs disjoint from calibration IDs | PASS (Test A + metadata) |
| No `validation_data` in freeze path | PASS (run_m3_freeze.py grepped) |
| Outer-eval engines absent from inner control | PASS (Test B) |
| CV matrix complete 8Ã—5 | PASS (assert_cv_complete, Test C) |
| Selection policy mechanically applied, decision recorded | PASS (selection_decision.json, Test D) |
| Freeze consumes YAML (no duplicate constants) | PASS (Test E; run_m3_freeze.py) |
| Canonical hashes platform-independent | PASS (Test F) |
| Conformal: exactly 15 scores, q table, limited wording | PASS (Test G; quantiles CSV) |
| FD004 fit IDs âŠ† allowed training IDs | PASS (Test H) |
| Sensitivity targets the M3 model, no SHAP language | PASS (explain_m3_sensitivity.py outputs) |
| Serving reads YAML + recalibrated quantiles | PASS (m1_predictor.py, app smoke) |
| Metadata per run (commit, seeds, hashes, versions) | PASS (fd001_outer_metadata.jsonl) |
| Post-hoc after calibration, recomputed from CSVs | PASS (falsification in run_m3_posthoc.py) |
| Docs M3-first, stale claims gone | PASS (README/PROJECT_SPEC/CHANGELOG) |

## Self-audit round 2 â€” live-doc falsification grep (2026-08-17)

- `"93 tests"` / `"78 tests"` as live counts: NOT FOUND in README/PROJECT_SPEC (historical-only, removed).
- `"8 candidates x 5 folds"` as a M2 claim: replaced by 40/40 gated matrix statements.
- `"winner"` (singular, non-policy): replaced by role-based champions; final report uses policy language.
- `q = 70.34` as current: removed (historical M2; current q(0.1) = 66.21).
- `"coverage 98%"` as current: replaced by 99% (Î±=0.1) post-hoc empirical + limited wording.
- `OOD` / `out-of-distribution`: absent from app and current docs (grep verified).
- `SHAP`: absent from M3 sensitivity artifacts (occlusion terminology only).
- Test counts verified live: full 134, artifact-free 130, artifact-gated 4 (needs_artifacts).

## Exit checklist (final; updated only when verified)

| # | Criterion | Status |
|---|---|---|
| 1 | .venv-only environment used | DONE |
| 2 | Git history intact; no rewrite | DONE |
| 3 | M3_REPAIR_PLAN.md exists with pre-registered policy | DONE |
| 4 | â‰¥10 critical issues fixed | DONE (16 issues, 5 CRITICAL) |
| 5 | Calibration leakage fixed in final FD001 fit | DONE (M3-1) |
| 6 | Outer-fold isolation (nested CV) | DONE (M3-2) |
| 7 | Complete 40/40 CV matrix with gate | DONE (M3-3) |
| 8 | Pre-registered selection policy applied | DONE (M3-4) |
| 9 | Conformal recalibrated on clean model | DONE (M3-5) |
| 10 | YAML-driven configs as source of truth | DONE (M3-6) |
| 11 | FD004 clean protocol | DONE (M3-8) |
| 12 | Sensitivity on M3 model | DONE (M3-9) |
| 13 | Docs updated (README/PROJECT_SPEC/CHANGELOG) | DONE (M3-10) |
| 14 | Protocol tests Aâ€“H, artifact-free | DONE (M3-11) |
| 15 | Falsification pass (metrics recomputed from CSVs) | DONE (M3-13) |
| 16 | Self-audit rounds 1+2 | DONE |
| 17 | Final report written | DONE (reports/m3_final_report.md) |
| 18 | CI artifact-free subset verified locally | DONE (130 passed) |
| 19 | App import smoke | DONE |
| 20 | Full test suite passes | DONE (134 passed) |

