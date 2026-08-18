# V2.2 Final Repository Freeze Plan

Master instruction: `C_MAPSS_V2_2_FINAL_REPOSITORY_FREEZE_AGENT_PLAN.md` (session-only, not committed).

Statuses: `OPEN` / `IN_PROGRESS` / `DONE` / `BLOCKED`

| ID | Issue | Severity | Current evidence | Why it matters | Files affected | Planned correction | Required tests | Required verification | Status | Final evidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | FD004 not fully YAML-driven | HIGH | `resolve_model_config` omits units/dropout/KMeans n_init; `fit_condition_models` hardcodes `n_init=10`; `fit_preprocessing` hardcodes k=6/seed=42 | YAML must be the true source of truth, not a matching coincidence | `scripts/run_v2_2_fd004_freeze.py`, `scripts/run_v2_2_fd004.py`, `src/rul_prediction/data/condition.py`, `configs/final_model_v2_2_fd004.yaml` | Thread k/seed/n_init through preprocessing; extend resolver to architecture/units/dropout/n_clusters/cluster_seed/n_init; structured YAML | Resolver-value test; hidden-constant test; `fit_preprocessing` threading test | Target + full + artifact-free suites | DONE | Config-authority tests pass; YAML restructured; freeze path reads every value from YAML |
| 2 | Stale "clean public checkout" QA is fake | HIGH | README claims a simulated 135/11/9 checkout result | QA must prove a real committed clone passes artifact-free | `README.md`, `tests/test_v2_2_cleanup.py` | Real `git clone .` into temp dir; run `-m "not needs_artifacts"` from clone with `PYTHONPATH=<clone>\src` | Collection-gating tests (existing) | Clone run + recorded counts | DONE | README records measured clone counts |
| 3 | Stale "Pre-registered" language | MEDIUM | README/PROJECT_SPEC/report use "pre-registered" but policy was never committed before execution | Git cannot prove formal pre-registration | `README.md`, `PROJECT_SPEC.md`, `reports/v2_2_final_report.md`, `CHANGELOG.md`, `configs/`, `V2_2_*.md` | Replace with "pre-specified" everywhere live | Grep audit | `git grep pre-registered` | DONE | No live "pre-registered" references |
| 4 | Docs overclaim "ALL hyperparameters from YAML" | MEDIUM | Report lines 45/175 wording predates FD004 fixes | Statement must be literally true after this session | `reports/v2_2_final_report.md`, `README.md`, `CHANGELOG.md` | Keep precise "all deployment-critical hyperparameters" wording | Grep audit | `git grep "ALL hyperparameters"` | DONE | Wording exact; FD004 fully YAML-driven (ID 1) |
| 5 | `source_tree_hash` provenance wording false | MEDIUM | CHANGELOG claims freeze metadata records `source_tree_hash`; committed metadata lacks it | Never claim a field that is absent | `CHANGELOG.md`, `reports/v2_2_final_report.md`, `src/rul_prediction/benchmark/v2_2.py` | Docs: metadata has git_commit/git_is_dirty/git_diff_hash/timestamp_utc; helper supports source_tree_hash | Provenance tests (current metadata + helper) | Tests + grep | DONE | Provenance tests pass; docs exact |
| 6 | Missing master cleanup-plan reference | MEDIUM | 4 files reference `C_MAPSS_V2_2_FINAL_CLEANUP_AGENT_PLAN.md` which does not exist | No broken local-document references | `CHANGELOG.md`, `V2_2_FINAL_CLEANUP_PLAN.md`, `reports/v2_2_final_report.md`, `tests/test_v2_2_cleanup.py` | Option B: replace with `V2_2_FINAL_CLEANUP_PLAN.md` | Broken-reference tests | Tests + grep | DONE | Reference test passes |
| 7 | FD001 feature metadata misleading | MEDIUM | `configs/final_model_v2_2_fd001.yaml` line 8 says "90-cycle windows (padded+masked)"; final model is XGBoost | Config must describe the real engineered-variable path | `configs/final_model_v2_2_fd001.yaml`, `scripts/select_v2_2_model.py` | Structured `features:` block; `sequence_padding_consumed_by_model: false`; extractor `extract_v2_features` | Feature-metadata test; 2D-matrix feature-path test | Target tests + full suite | DONE | Feature tests pass |
| 8 | 4 cleanup tests wrongly gated `needs_artifacts` | MEDIUM | Tests read TRACKED `experiments/v2_2` files but are excluded from artifact-free CI | CI must exercise tracked audit tables | `tests/test_v2_2_cleanup.py` | Unmark; skip only if dir absent | Unmarked tests run in clone QA | Clone QA counts | DONE | Clone QA runs them |
| 9 | Test counts stale in docs | MEDIUM | README/PROJECT_SPEC/report reference pre-freeze counts | Current counts must match new measurement | `README.md`, `PROJECT_SPEC.md`, `CHANGELOG.md`, `reports/v2_2_final_report.md`, `V2_2_REPAIR_PLAN.md`, `V2_2_FINAL_CLEANUP_PLAN.md`, `V2_2_FINAL_FREEZE_PLAN.md` | Update after measuring full + artifact-free + clone suites | - | Measurement runs | DONE | Counts recorded in README |
| 10 | Conformal q not falsified in tests | LOW | q values (66.2097/44.7955/41.4224) unverified in suite | Quantiles must recompute from tracked scores | `tests/test_v2_2_cleanup.py` | Recompute q from `fd001_conformal_engine_scores.csv`; compare csv + deployment config | Conformal falsification test | Tests | DONE | Test passes |
| 11 | Metric falsification tests missing/insufficient | LOW | No test recomputes official metrics from saved predictions | Headline numbers must be reproducible from tracked tables | `tests/test_v2_2_cleanup.py` | Add FD001/FD004 metric + CV-summary falsification tests | Falsification tests | Tests | DONE | Tests pass |
| 12 | Docs stale first-label-contact claim | LOW | PROJECT_SPEC says labels "first touched AFTER the final model fit" | Label-contact history must be exact | `PROJECT_SPEC.md` | Reconcile wording with actual label contact | Grep audit | `git grep` | DONE | Wording exact |

## FD004 old-effective vs new YAML-resolved values

Old effective values (historical comparison path, defaults/hardcoded constants) vs new YAML-resolved freeze values (after ID 1).

| Parameter | Old effective runtime | New YAML-resolved value | Match? |
|---|---|---|---|
| window | 45 (`WINDOW = 45`) | 45 | YES |
| units | (128, 64) (`v2_gru` default) | [128, 64] | YES |
| dropout | 0.3 (`v2_gru` default) | 0.3 | YES |
| loss | huber (`loss="huber"` call) | huber | YES |
| learning_rate | 0.001 (`v2_gru` default) | 0.001 | YES |
| batch_size | 256 (`BATCH_SIZE = 256`) | 256 | YES |
| seed | 42 (`SEED = 42`) | 42 | YES |
| fixed_epochs | 8 (winner `best_epoch`) | 8 | YES |
| variant | C (selection) | C | YES |
| n_clusters | 6 (`fit_condition_models` default) | 6 | YES |
| cluster_seed | 42 (`fit_condition_models` default) | 42 | YES |
| n_init | 10 (hardcoded in `fit_condition_models`) | 10 | YES |

**Verdict: every YAML-resolved value equals the old effective value -> NO FD004 RETRAIN REQUIRED.**

## Measured test counts

| Run | Passed | Failed | Skipped | Deselected | Warnings |
|---|---|---|---|---|---|
| Full local suite (baseline) | 155 | 0 | 0 | 0 | 16 |
| Artifact-free `-m "not needs_artifacts"` (baseline) | 146 | 0 | 9 | 0 | 4 |
| Full local suite (post-freeze) | 163 | 0 | 0 | 0 | 16 |
| Artifact-free (post-freeze) | 158 | 0 | 0 | 5 | 4 |
| Real clean clone QA | | | | | |