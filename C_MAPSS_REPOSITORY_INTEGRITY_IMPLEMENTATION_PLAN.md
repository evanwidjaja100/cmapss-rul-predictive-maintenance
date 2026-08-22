# C-MAPSS Repository Integrity and Reproducibility Implementation Plan

**Plan status:** READY FOR IMPLEMENTATION

**Prepared:** 2026-08-21

**Repository baseline:** `main` at `6151773`
**Primary objective:** complete the repository-integrity, configuration-authority,
provenance, artifact-verification, CI, application, test-architecture, dependency,
and documentation repairs without changing the frozen scientific results.

---

## 1. Execution Directive

The implementing AI agent must treat this document as an executable plan and live
tracker. It must use sub-agents, maintain evidence for every status change, and
continue through the verification loops until every applicable exit criterion is
green.

The agent must not declare completion merely because targeted tests pass. Completion
requires the full local suite, artifact-free suite, repository-integrity checks,
artifact verification, dependency checks, application smoke checks, and a real
clean-clone verification against the final committed state.

If the agent does not have commit authorization, it must complete every pre-commit
gate and report the final clean-clone gate as pending. It must not simulate a clean
clone from uncommitted files or describe such a simulation as clean-clone QA.

---

## 2. Mission and Goals

### Goal G1 - Restore the audit trail

Restore the four deleted historical plan files exactly from Git history, make all
live repository-local references resolvable, and preserve the distinction between
historical records and current behavior.

### Goal G2 - Make FD004 configuration behaviorally authoritative

Create one typed and validated FD004 configuration contract shared by config
generation, final freeze, artifact naming, preprocessing persistence, and post-hoc
inference. Every accepted behavior-driving configuration value must reach the
corresponding runtime operation; otherwise the configuration must fail before any
training, loading, or output write.

### Goal G3 - Make provenance deterministic and honest

Replace the current filesystem-recursive source hash with a versioned Git-tracked
execution-input hash. Define staged, unstaged, untracked, ignored, and dirty-run
semantics precisely. Future freezes must reject dirty execution inputs by default,
while unrelated non-execution files may remain present if whole-repository dirtiness
is recorded honestly. Future freeze runs must capture provenance before training.
Historical metadata must remain historical and must not be retroactively upgraded.

### Goal G4 - Create verifiable artifact lineage

Add checked-in FD001 and FD004 artifact manifests connecting configurations, split
manifests, models, preprocessors, predictions, metrics, selection evidence,
conformal evidence, metadata, and dependency constraints. Verification must work in
both tracked-only clean-clone mode and full local artifact mode.

### Goal G5 - Make current correctness come from CI, not mutable prose

Remove permanent numeric "current test count" claims from README and PROJECT_SPEC.
Preserve dated counts in historical reports and changelog entries, tied to a date and
commit. Current branch health must be represented by CI and reproducible commands.

### Goal G6 - Generalize reference and text-integrity checking

Replace the hand-maintained reference test with a Git-tracked repository-integrity
checker. It must detect missing or ambiguous local references, invalid UTF-8,
replacement characters, and known mojibake sequences.

### Goal G7 - Make the Streamlit application import-safe

`import app_v2` must not instantiate a predictor, load a model or scaler, read raw
data, initialize TensorFlow for an XGBoost deployment, render Streamlit UI, or emit
bare-Streamlit warnings. Real Streamlit execution must retain current behavior.

### Goal G8 - Introduce meaningful test tiers

Every test must have one primary tier. Tests that require gitignored data, models, or
generated artifacts must also carry `needs_artifacts`. Tests reading committed
`experiments/v2_2` evidence must continue to run in CI and in a clean clone.

### Goal G9 - Align dependencies and CI

CI must consume validated pinned constraints, run `pip check`, and use a declared
Python version consistent with the reproducibility documentation. Do not install the
existing Windows freeze directly as a Linux requirements file without validating
platform compatibility.

### Goal G10 - Produce a durable implementation record

Update this plan's tracker and create a final implementation report containing
changed files, commands, test results, model-hash preservation evidence, scientific
falsification results, limitations, and the exact commit verified in a clean clone.

---

## 3. Non-Negotiable Scientific and Repository Constraints

1. Do not rerun the full FD001 cross-validation matrix.
2. Do not rerun FD004 variants A/B/C/D.
3. Do not retrain the FD001 or FD004 frozen models merely to improve metadata,
   configuration plumbing, manifests, tests, or documentation.
4. Do not change the selected FD001 deployment model `xgb_w90_d6`.
5. Do not change the selected FD004 variant C.
6. Do not change current headline metrics, conformal scores, quantiles, predictions,
   split membership, or selection decisions.
7. Do not rewrite historical metadata to imply that new provenance fields existed at
   the time of the historical training run.
8. Do not commit `data/raw`, `data/processed`, `.venv`, caches, or model binaries.
9. Do not delete, edit, or commit the user's untracked `session-ses_feb9.md` unless
   the user separately requests it.
10. Do not rewrite Git history.
11. Preserve unrelated user changes if the worktree becomes dirty during execution.
12. Use explicit validation exceptions for external configuration and manifest
    errors. Do not rely on `assert` for contracts that must survive `python -O`.
13. Any present-but-tampered model or preprocessor must fail closed before
    deserialization. Missing local artifacts must retain a separate friendly error.
14. Do not weaken a test, add a broad skip, or expand an allowlist merely to make CI
    green.

---

## 4. Confirmed Baseline and Preservation Ledger

The implementation agent must independently recheck this baseline before editing.

### 4.1 Current repository state

- Baseline HEAD: `6151773`.
- Intentional untracked implementation artifact at plan-authoring time:
  `C_MAPSS_REPOSITORY_INTEGRITY_IMPLEMENTATION_PLAN.md`. The lead agent may edit,
  stage, and commit this plan as part of the implementation.
- User-owned untracked file: `session-ses_feb9.md`.
- Commit `6151773` deleted:
  - `V2_1_REPAIR_PLAN.md`
  - `V2_2_REPAIR_PLAN.md`
  - `V2_2_FINAL_CLEANUP_PLAN.md`
  - `V2_2_FINAL_FREEZE_PLAN.md`
- All four files exist at commit `23cc934`.
- Current full test result: 162 passed, 2 failed, 16 warnings.
- Current artifact-free local result: 157 passed, 2 failed, 5 deselected,
  4 warnings.
- Current real clean-clone result: 147 passed, 10 skipped, 5 deselected,
  2 failed.
- Both failures are caused by the deleted plans.

### 4.2 Frozen binary preservation hashes

These hashes are immutable gates for this repair unless the user separately
authorizes retraining:

| Artifact | Baseline SHA-256 |
|---|---|
| FD001 frozen model | `23bd460cd447141d90fd045b863f1a33db02845b83ce9d4148791811ad5a9d6b` |
| FD001 scaler | `d1b02cfc7043de8e68f9dd71bc1a99efbcd2754a921a59524a431647d0371a48` |
| FD004 frozen model | `9b39a059ea528b94f9d806533a69da596b7950254bbea1f11a232b2bb29ac87d` |
| FD004 condition preprocessor | `f22ef7189cda906ee4ea92c37df625023f136abb3fcd8a0a74e3a0fdf8b4a328` |

### 4.3 Scientific preservation facts

- FD001 CV has 40 candidate-fold rows: eight candidates, folds 1-5.
- FD001 deployment selection: `xgb_w90_d6`.
- FD001 accuracy champion: `lstm_w60_huber`.
- FD001 NASA-risk champion: `xgb_w90_d6`.
- FD004 selection: variant C.
- Conformal calibration has 15 engine scores.
- Stored conformal q values are 66.2097, 44.7955, and 41.4224 for alpha
  0.1, 0.2, and 0.3 respectively.
- Official FD001 and FD004 metrics must continue to recompute from saved
  prediction tables within existing tolerances.

### 4.4 Baseline evidence file

Before implementation, create
`reports/repository_integrity_preservation_ledger.json`. It is the single
machine-readable preservation authority. Use a versioned schema with these required
top-level objects: `capture`, `repository`, `binaries`, `configs`,
`scientific_evidence`, `tests`, `environment`, and `tracked_experiment_files`.
Include at least:

- baseline commit and `git status --porcelain=v2`;
- the four binary hashes above;
- hashes of the FD001 and FD004 final configs;
- hashes of selection, conformal, official prediction, and final metric files;
- the current test results;
- Python and the six core ML package versions;
- the list of tracked `experiments/v2_2` files.

Each hashed entry must contain repository-relative path, SHA-256, byte size, and
existence/storage status. Each test entry must contain command, exit code, counts,
and timestamp. The ledger must clearly say it was captured at repair time, not
training time. Validate it against a checked-in JSON Schema or a strict equivalent
validator. Render a human-readable summary from this JSON into the final
implementation report; do not maintain a second hand-edited preservation table.

---

## 5. Issue Tracker

Statuses are `OPEN`, `IN_PROGRESS`, `BLOCKED`, or `DONE`. The implementing agent
must update this table with a concise evidence reference before final handoff.

| ID | Severity | Issue | Primary owner | Status | Completion evidence |
|---|---|---|---|---|---|
| RI-01 | Critical | Four audit/repair plans deleted; current CI fails | History/integrity agent | DONE | 4 files restored byte-identical to `23cc934` blobs (`f5c1e79`, `2631a59`, `bbd55cd`, `2b82d83`); both baseline reference tests green; integrity checker OK (14 anchors) |
| RI-02 | Critical | FD004 architecture is read but ignored; inference window remains hardcoded | FD004 agent | DONE | `fd004_config.py` gru-only fail-closed; `make_predictor(*, window)` keyword-only in FD004 + shared benchmark path (14 callers updated); window=47 tensor/mask test |
| RI-03 | High | FD004 config does not validate optimizer, clustering method, split paths, identity, or payload compatibility | FD004 agent | DONE | Typed contract validates optimizer `{name, clipnorm}`, KMeans-only, split paths/counts/canonical hashes/disjointness + RAW file hashes, candidate identity, payload schema/identity/dims |
| RI-04 | High | Source hash includes ignored/generated files and is not fail-closed | Provenance agent | DONE | `tracked_source_tree_details` via `git ls-files -z` only; `cmapss-tracked-source-v1` length-delimited; worktree bytes hashed; read failures raise (29 falsification tests) |
| RI-05 | High | Dirty/staged/untracked provenance is incomplete; freeze provenance is captured too late | Provenance agent | DONE | Whole-vs-execution dirty flags, staged/unstaged/untracked-dir expansion, snapshot policy with traversal/symlink/size/secret guards; `assert_reproducible_run_state` wired into variant-run/freeze/posthoc before training/loading |
| RI-06 | High | Frozen binaries have no checked-in integrity anchor or full lineage manifest | Manifest agent | DONE | `fd001_artifact_manifest.json` (25 artifacts), `fd004_artifact_manifest.json` (14); tracked+full verification green; deterministic `--check`; tamper tests via temp root override |
| RI-07 | High | Permanent "current" test counts are stale after later commits | History/integrity agent | DONE | README/PROJECT_SPEC carry no mutable current counts; dated snapshot at `23cc934` in CHANGELOG; `test_documentation_truth.py` guard |
| RI-08 | High | Local-reference checking is narrow and hand-maintained | History/integrity agent | DONE | `scripts/check_repository_integrity.py`: git-tracked enumeration, root/dir/basename resolution, unused-exception failure, encoding checks; 7 narrow used exceptions |
| RI-09 | Medium | `app_v2` performs model/UI/data work at import and initializes TensorFlow unnecessarily | App agent | DONE | Import side-effect free (subprocess tests), TF lazy in neural branch only, headless smoke `app_smoke+needs_artifacts` green with traceback scan |
| RI-10 | Medium | Tracked text contains mojibake/replacement-character corruption | History/integrity agent | DONE | Encoding checker rejects U+FFFD + 11 mojibake sequences; fixes applied in CHANGELOG/AUDIT_V2/tests; byte-frozen historical plan excepted with justified entry |
| RI-11 | Medium | Registered unit/integration markers are unused; artifact needs are inconsistently expressed | CI/test agent | DONE | 6 markers registered; conftest enforces exactly-one-primary-marker; `test_marker_audit.py`; nested-collection test removed |
| RI-12 | High | CI ignores pinned constraints and Python/environment wording drifts | CI/dependency agent | DONE | CI pins Python 3.12.10 + pip 26.2.1, platform-selected constraints via `-c`, `pip check`; dependency-consistency checker; PROJECT_SPEC wording pinned to 3.12.10 |
| RI-13 | Medium | FD004 partial/resumed runs can produce incomplete config or overwrite best-epoch rows | FD004 agent | DONE | Canonical write refused unless all A/B/C/D rows (resolved-path guard); best-epoch merge collision raises; duplicates rejected; 37×5 prediction requirement enforced |
| RI-14 | Medium | FD004 lacks a canonical experiment-side official-prediction file | Manifest agent | DONE | `experiments/v2_2/fd004_official_predictions.csv` byte-identical to report mirror; manifest enforces hash equality; `--check` never creates it |

---

## 6. Mandatory Sub-Agent Strategy

The lead agent must use sub-agents. With four concurrent slots, use the lead agent
plus at most three workers at once. Reuse completed worker slots for later review
rounds.

### 6.1 First implementation wave

#### Sub-agent A - History, references, documentation integrity

Owns:

- the four restored plan files;
- `scripts/check_repository_integrity.py`;
- `configs/repository_integrity.yaml`, if exceptions are genuinely required;
- `tests/test_repository_integrity.py`;
- mojibake corrections in `CHANGELOG.md`, `AUDIT_V2.md`, and affected tests.

It must not edit FD004 runtime code, provenance code, dependency files, or the
Streamlit application.

#### Sub-agent B - FD004 configuration authority

Owns:

- the shared FD004 config module;
- `configs/final_model_v2_2_fd004.yaml`;
- `scripts/run_v2_2_fd004.py`;
- `scripts/run_v2_2_fd004_freeze.py`;
- `scripts/run_v2_2_fd004_posthoc.py`;
- FD004-related changes in `src/rul_prediction/data/condition.py` and
  `src/rul_prediction/models/v2_models.py`;
- focused FD004 configuration and round-trip tests.

It must not retrain, rerun variants, edit tracked results, or alter historical
metrics.

#### Sub-agent C - Provenance and source hashing

Owns:

- the new reproducibility module;
- compatibility re-exports in `benchmark/v2_2.py`;
- provenance unit and falsification tests.

It must not edit freeze scripts concurrently with Sub-agent B. It should provide a
stable interface that the lead agent integrates after FD004 changes settle.

### 6.2 Second implementation wave

After the first wave is integrated, reuse sub-agent slots for:

#### Sub-agent D - Artifact manifests

Owns the manifest library, builder/verifier scripts, two V2.2 manifest files,
canonical FD004 official predictions, and manifest tests.

#### Sub-agent E - CI, dependency constraints, and test taxonomy

Owns `.github/workflows/ci.yml`, dependency files, dependency-consistency checks,
`pyproject.toml` markers, and marker classification. It must coordinate marker names
with the owners of new tests rather than editing the same new test files
concurrently.

#### Sub-agent F - Import-safe application

Owns `app_v2.py`, lazy serving imports in `v2_predictor.py`, and app smoke tests.

### 6.3 Required independent review wave

After implementation, spawn or reuse at least two sub-agents as read-only reviewers:

1. A reproducibility/adversarial reviewer must attack source-hash semantics, dirty
   snapshots, path traversal, manifest tampering, historical provenance wording,
   and accidental binary changes.
2. A repository/CI reviewer must attack broken references, test classification,
   documentation truth, clean-clone behavior, dependency constraints, app import
   side effects, and encoding.

The lead agent must fix all P0/P1/P2 findings and rerun affected loops. A final review
is clean only when both reviewers independently report no remaining material
findings.

### 6.4 Coordination rules

- Every sub-agent reads the latest `git status` and relevant files before editing.
- Assign disjoint file ownership before concurrent work.
- Only the lead agent edits cross-cutting documentation such as README,
  PROJECT_SPEC, the final report, and the final implementation report after worker
  integration.
- Only the lead agent creates commits unless the user explicitly requests another
  workflow.
- Every sub-agent reports changed files, exact commands, results, unresolved risks,
  and any assumptions.
- The lead agent reviews diffs directly; "tests passed" is not sufficient evidence.
- If a worker needs a file owned by another active worker, it sends an interface
  request and waits rather than editing concurrently.

---

## 7. Phase 0 - Preflight and Baseline Capture

### Actions

1. Read any repository `AGENTS.md` files if they appear after this plan was written.
2. Run `git status --short --branch` and preserve all pre-existing changes.
3. Confirm `session-ses_feb9.md` remains untracked and untouched.
4. Confirm each deleted plan exists in `23cc934` and record its historical blob ID.
5. Recompute the four frozen binary hashes.
6. Run the current focused cleanup suite, full suite, and artifact-free suite.
7. Recompute selection, conformal q, FD001 official metrics, FD004 official metrics,
   and CV summaries from saved tables.
8. Record baseline dependency and Python versions.
9. Create/update the preservation ledger and set RI-01 through RI-14 to their
   confirmed statuses.

### Baseline commands

```powershell
git status --short --branch
git log --oneline -8
git ls-files experiments/v2_2
.\.venv\Scripts\python.exe -m pytest tests\test_v2_2_cleanup.py -q
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest -q -m "not needs_artifacts"
```

### Gate P0

Do not begin implementation unless the lead agent can distinguish pre-existing user
changes from planned changes and has captured the preservation hashes. If a frozen
binary already differs from the values in Section 4, stop and report it before
continuing.

---

## 8. Phase 1 - Restore and Protect Historical Plans

### Actions

1. Restore exactly these paths from commit `23cc934`:
   - `V2_1_REPAIR_PLAN.md`
   - `V2_2_REPAIR_PLAN.md`
   - `V2_2_FINAL_CLEANUP_PLAN.md`
   - `V2_2_FINAL_FREEZE_PLAN.md`
2. Do not restore obsolete continuation checklists or session-only master agent
   instructions.
3. Restore all four files byte-for-byte and do not annotate or edit them. Put any
   snapshot clarification in README and the final implementation report.
4. Verify each restored working-tree file against `23cc934:<path>` with
   `git hash-object`; the working-tree blob ID must equal the historical blob ID.
5. Run the two previously failing required-reference tests immediately.
6. Search all tracked docs and source for references to the four plans and verify
   that every live reference now resolves.

### Verification

```powershell
git diff --name-status
git rev-parse '23cc934:V2_1_REPAIR_PLAN.md'
git hash-object V2_1_REPAIR_PLAN.md
git rev-parse '23cc934:V2_2_REPAIR_PLAN.md'
git hash-object V2_2_REPAIR_PLAN.md
git rev-parse '23cc934:V2_2_FINAL_CLEANUP_PLAN.md'
git hash-object V2_2_FINAL_CLEANUP_PLAN.md
git rev-parse '23cc934:V2_2_FINAL_FREEZE_PLAN.md'
git hash-object V2_2_FINAL_FREEZE_PLAN.md
.\.venv\Scripts\python.exe -m pytest tests\test_v2_2_cleanup.py -q `
  -k "required_tracked_local_references or broken_master_cleanup"
```

After the lead agent stages or commits the restored files, use
`git ls-files --error-unmatch <path>` to verify tracked state. Do not use that
command as the pre-staging content check.

### Gate P1

- All four files exist and are tracked.
- No unrelated historical file is restored.
- The two baseline test failures are gone.
- No historical claim is silently rewritten as if it were contemporary.

---

## 9. Phase 2 - FD004 Authoritative Configuration Contract

### 9.1 Design decision

Keep V2.2 FD004 deployment explicitly GRU-only in this repair. Accept
`architecture: gru`; reject any other architecture before side effects. Do not
pretend that unselected LSTM or TCN deployment is supported.

Prefer an immutable dataclass-based implementation to avoid a new runtime schema
dependency. A suitable location is:

- `src/rul_prediction/benchmark/fd004_config.py`

Provide:

- `FD004FinalConfig`;
- `from_mapping(mapping)`;
- `load_fd004_final_config(path)`;
- canonical config hashing;
- model/preprocessor artifact path generation;
- repository-root-relative split and validation-manifest path resolution;
- candidate identity validation.

### 9.2 Behavior-driving fields

The shared contract must include and validate:

- methodology version and dataset;
- candidate name and architecture;
- window;
- exactly two GRU unit counts;
- dropout;
- loss;
- optimizer name and clipnorm;
- learning rate;
- batch size;
- model seed;
- fixed epochs;
- selected variant;
- clustering method, cluster count, clustering seed, and `n_init`;
- operating-setting columns;
- sensor-scaling mode and fit scope;
- split provenance path;
- validation cutoff-manifest path;
- expected development, validation, calibration, final-fit, and reserved counts;
- expected engine-ID hashes;
- final-fit validation-data rule.

Use structured optimizer YAML:

```yaml
optimizer:
  name: adam
  clipnorm: 1.0
```

If the old optimizer string must be read for compatibility, accept only its exact
known value, normalize it internally, and always write the structured format.

The structured YAML is a post-training normalization for future reproducibility; it
is not the literal file used to train the historical frozen FD004 model. Record an
old-effective-versus-new-resolved mapping in the preservation ledger and final
report. The artifact manifest must label the new YAML as post-training normalized
and behaviorally equivalent, and must never present its raw byte hash as the
historical training-config hash.

Define two non-interchangeable config identity fields:

- `config_file_sha256`: SHA-256 of exact raw file bytes, used for artifact
  integrity;
- `config_canonical_sha256`: SHA-256 of the normalized typed mapping using an
  explicitly named, versioned canonicalization algorithm, used for semantic
  identity.

Apply the same distinction to split evidence: use raw SHA-256 fields for exact split
and cutoff files, while retaining separately named canonical engine-ID/manifest
hashes for semantic membership validation. Never compare a raw-file digest to a
canonical-set digest.

### 9.3 Fail-closed validation

Reject before side effects:

- methodology other than 2.2 or dataset other than FD004;
- architecture other than GRU;
- clustering method other than KMeans;
- candidate identity inconsistent with architecture/window/loss/variant;
- units not containing exactly two positive, non-boolean integers;
- nonpositive window, batch size, epochs, clusters, or `n_init`;
- non-finite or invalid dropout/learning rate;
- unsupported loss or optimizer configuration;
- operating-setting columns or scaling contract inconsistent with implementation;
- `validation_data_in_final_fit: true`;
- split counts/hashes/disjointness inconsistent with YAML;
- selected-variant best epoch inconsistent with `fixed_epochs`.

### 9.4 Runtime propagation

Update `run_v2_2_fd004_freeze.py` so all behavior is delegated through testable
functions, such as:

- `load_and_validate_split(config, frame)`;
- `fit_fd004_final_model(config, frame, split_plan)`;
- `save_fd004_final_artifacts(config, model, preprocessing, metadata)`.

Required behavior:

- load split and cutoff paths named by the validated config, not hardcoded CWD paths;
- recompute split hashes and counts;
- verify all IDs exist and all sets are disjoint;
- fit the final model/preprocessing on development plus validation IDs only;
- keep calibration IDs reserved;
- pass window, units, dropout, loss, optimizer, clipnorm, learning rate, seed,
  batch size, epochs, cluster count, cluster seed, and `n_init` explicitly;
- never pass validation data to final `model.fit`;
- derive artifact names from the validated identity;
- persist a versioned preprocessing payload containing candidate, variant, config
  hashes, fit-ID hash, preprocessing type, and every required preprocessing object
  for future freezes only;
- include `global_scaler` for A/B and condition objects for C/D, or explicitly reject
  unsupported variants before training;
- anchor paths under repository root;
- write metadata atomically;
- never import or call official-label loading in the freeze path.

### 9.5 Shared inference path

Change `make_predictor` to require a keyword-only `window` and use that value for
both `build_window` and `window_mask`.

Update FD004 post-hoc evaluation to:

- load the same typed config;
- resolve identical artifact paths;
- load and verify the versioned preprocessing payload for future-schema artifacts,
  or use the narrowly defined historical adapter below;
- pass `window=config.window` explicitly;
- verify Keras time and feature dimensions against config/transformed input;
- distinguish missing, legacy-incompatible, and hash-mismatched artifacts;
- read official labels only after model/config/preprocessor compatibility passes;
- keep official metrics permanently post-hoc and selection-inert.

### 9.5.1 Legacy FD004 preprocessing payload rule

Do not rewrite `fd004_conditionC.joblib`; its baseline SHA-256 is an immutable gate.
The new versioned payload format applies only to future freezes. Current post-hoc
loading may enter a narrowly scoped legacy adapter only when all of these are true:

- the joblib SHA-256 equals the Section 4 baseline and manifest value;
- all historically required keys and preprocessing objects are present;
- sidecar/manifest identity resolves to variant C and the expected candidate;
- fitted feature counts, cluster dimensions, model input dimensions, and config
  window agree;
- no deserialization occurs before the file hash check.

Put any new integrity or semantic metadata in a checked-in sidecar/manifest; never
inject it into the historical joblib. Tests must cover a synthetic future-schema
payload and the hash-verified historical legacy payload path. A legacy payload with
an unknown hash, missing key, wrong identity, or incompatible dimension fails
closed.

### 9.6 Variant-run/config-writer consistency

Refactor `run_v2_2_fd004.py` so one explicit experiment recipe supplies values to
training and to config writing. Do not reconstruct the final YAML from scattered
literals.

Also fix these consistency risks without running the experiment:

- refuse to write the canonical final config unless all A/B/C/D result rows exist;
- merge resumed best-epoch rows by variant instead of overwriting prior rows;
- reject duplicates;
- require exactly 37 x 5 validation predictions for each completed variant;
- clarify selection stage 1 (150/25), selection stage 2 (175), final freeze (212),
  reserved calibration (37), and official-test transform-only scopes in YAML.

### 9.7 Behavioral tests

Add `tests/test_v2_2_fd004_config.py` or an equivalently focused module. Tests must
include:

1. Production YAML parses to the expected contract.
2. Table-driven invalid-config cases.
3. A deliberately non-default config reaches every preprocessing, sequence,
   builder, optimizer, fit, and metadata argument through spies/stubs.
4. Unsupported architecture fails before preprocessing/model construction/writes.
5. `window=47` produces a `(1, 47, features)` inference tensor and `(1, 47)` mask.
6. Freeze and post-hoc resolve identical artifact paths.
7. Temporary split paths named by YAML are consumed; overlap/count/hash drift fails.
8. Synthetic future-schema A/B and C/D preprocessing payloads round trip, or
   unsupported paths fail early; the verified historical C payload uses only the
   bounded legacy adapter and remains byte-identical.
9. Config, metadata, artifact name, payload, and model input shape cannot disagree.
10. Freeze cannot read official labels.
11. Resume/idempotency behavior preserves prior variant control rows.
12. Partial results cannot overwrite the canonical final config.

Literal-source scans may remain as secondary defense, but behavioral tests are the
authority.

### Gate P2

- Current numerical YAML values remain unchanged.
- No model or selection rerun occurs.
- Existing four model/preprocessor hashes remain unchanged.
- The historical FD004 joblib remains byte-identical; no migration rewrites it.
- Raw-file and canonical-semantic config/split hashes are separately named and
  tested.
- A non-45 synthetic window reaches training and inference behavior.
- Unsupported architecture/method fails before side effects.
- Targeted FD004, protocol, cleanup, and condition tests pass.
- Saved-result falsification still selects C and reproduces official metrics.

---

## 10. Phase 3 - Deterministic Provenance

### 10.1 New reproducibility module

Create `src/rul_prediction/reproducibility.py` or a small package with equivalent
ownership. Preserve compatibility imports in `benchmark/v2_2.py`.

Implement:

- streaming `sha256_file(path)`;
- Git root resolution;
- Git-tracked execution-input enumeration using `git ls-files -z`;
- `tracked_source_tree_details(root)`, whose canonical digest field remains
  `source_tree_hash`;
- `collect_git_provenance(...)`;
- `assert_reproducible_run_state(...)`.

Execution-input scope must be explicit and versioned, including at least:

- `src/**`
- `scripts/**`
- `configs/**`
- `app_v2.py`
- `.github/workflows/**`
- `pyproject.toml`
- `requirements.txt`
- `requirements-lock.txt`

Do not use recursive filesystem enumeration. Do not include `__pycache__`, `.pyc`,
`.egg-info`, models, raw data, pytest caches, or ignored generated files.

### 10.2 Hash encoding

Use a domain-separated, length-delimited format, for example
`cmapss-tracked-source-v1`. For each sorted POSIX path, encode a type tag, path
length, path bytes, content length, and raw content. Do not concatenate ambiguous
path/content streams. Do not silently skip read failures.

Return at least:

- hash;
- algorithm identifier;
- file count;
- normalized file list.

### 10.3 Dirty-state semantics

Record both whole-repository state and execution-scope state. Define:

- clean tracked source;
- staged tracked edits;
- unstaged tracked edits;
- tracked deletion/rename;
- relevant untracked execution file;
- unrelated untracked documentation/session file;
- ignored generated file.

Use NUL-delimited Git status output and a binary diff representation that includes
staged and unstaged changes. Relevant untracked execution files must be content
hashed. Ignored caches must not affect execution hashes.

Future freeze policy:

- reject dirty execution inputs by default before training;
- allow unrelated non-execution files without a dirty-source bundle, but record the
  whole-repository status separately from the clean execution scope;
- permit dirty execution only with an explicit flag, nonempty reason, and durable
  caller-supplied snapshot destination;
- store a binary patch, exact copies of relevant untracked execution files, a
  path/size/hash inventory, and a snapshot hash;
- reject path traversal, repository-escaping symlinks, unreadable inputs, or an
  incomplete snapshot.

Dirty-snapshot safety is mandatory: never invent a destination, never auto-stage or
auto-commit snapshot contents, reject oversized files using a documented limit,
and scan/report likely secrets or sensitive filenames before copying. If a snapshot
is ever proposed for tracked evidence, stop and obtain explicit user confirmation
after showing the destination and sensitive-data risk summary. Snapshot tests must
use temporary repositories and synthetic non-secret fixtures.

The user's untracked session transcript is not an execution input, but whole-tree
dirty status may still record its presence honestly.

### 10.4 Provenance schema

Future metadata should contain:

- schema version;
- run-start timestamp;
- Git commit;
- whole-tree and execution-tree dirty flags;
- Git status hash;
- diff hash;
- canonical `source_tree_hash`, algorithm, and file count;
- execution source hash;
- relevant untracked input list;
- dirty reason and snapshot path/hash when used;
- exact config path, `config_file_sha256`, and versioned
  `config_canonical_sha256`;
- split/cutoff paths with raw file hashes, plus separately named canonical
  membership/engine-ID hashes;
- constraints path/hash;
- Python executable/version and package versions.

Capture this at run start before fitting or output writes. After saving, add
model/preprocessor path, size, and hash, then write metadata atomically.

Do not alter historical FD001/FD004 metadata to imply this schema existed during
training.

### 10.5 Provenance falsification tests

Test clean tree, staged edit, unstaged edit, untracked source, ignored cache, tracked
deletion, rename, editable-install `.egg-info`, mtime change, enumeration order,
Unicode content, Windows/POSIX path normalization, missing/unreadable input, and
dirty-snapshot reconstruction in temporary repositories.

### Gate P3

- Source hashing uses only Git-tracked execution inputs.
- Imports, pytest, and editable install do not change it.
- Tracked content changes do change it.
- Dirty execution is rejected or durably captured.
- Docstrings exactly match behavior.
- No broad `except OSError: continue` remains.

---

## 11. Phase 4 - Artifact Manifests and Load-Time Verification

### 11.1 Deliverables

Create:

- `src/rul_prediction/artifact_manifest.py`;
- `scripts/build_v2_2_artifact_manifests.py`;
- `scripts/verify_v2_2_artifacts.py`;
- `experiments/v2_2/fd001_artifact_manifest.json`;
- `experiments/v2_2/fd004_artifact_manifest.json`;
- `experiments/v2_2/fd004_official_predictions.csv` derived from the existing
  tracked report table without model inference or retraining;
- focused manifest tests.

### 11.2 Manifest semantics

Each manifest must record:

- schema and methodology version;
- dataset and model ID;
- historical training-provenance status and limitations;
- repository-relative POSIX paths only;
- role, path, SHA-256, size, storage class, clean-clone requirement, and hash kind;
- explicit lineage connections;
- source/config/constraints integrity context captured at manifest-generation time.

Storage classes:

- `git`: must exist and verify in a clean clone;
- `local`: intentionally gitignored runtime artifact.

Verification modes:

- `--mode tracked`: every Git artifact is required; absent local binaries are
  permitted; a present wrong local binary fails.
- `--mode full`: every Git and local artifact is required and verified.

Builder/verifier operational contract:

- `--check` performs validation and deterministic regeneration comparison without
  rewriting any manifest;
- manifest creation accepts `--generated-at <UTC>` or preserves the prior
  `generated_at_utc` when all hashed inputs are unchanged;
- with identical inputs and a fixed timestamp, serialized JSON is byte-for-byte
  deterministic (stable ordering, formatting, newline, and UTF-8 encoding);
- library APIs and CLI accept an explicit repository/bundle `root` override;
- all tamper, missing-file, and wrong-file tests operate on temporary copied bundles
  through that override and never modify the real frozen artifacts.

No manifest may hash itself. No broad globs may define critical artifacts. Reject
absolute paths, `..`, duplicate roles/paths, wrong dataset/model identity, and
ambiguous mirrors.

### 11.3 Required FD001 lineage

Connect at least final config, deployment config, model, scaler, final metadata,
outer split manifest, five outer cutoff CSVs, calibration cutoffs, fold results,
outer predictions, engine-level results, CV summary, best iterations, selection
decision, conformal scores/quantiles/calibration, official predictions, final
metrics, and dependency constraints.

### 11.4 Required FD004 lineage

Connect at least final config, model, condition preprocessor, final metadata, split
JSON, validation cutoff CSV, variant results/predictions, best epochs, canonical
official predictions, report-table mirror, final metrics, and dependency constraints.

The canonical FD004 experiment prediction table must exactly equal the existing
report-table presentation mirror.

### 11.5 Historical migration wording

- FD001 training provenance remains `historical_dirty_partial` or an equally exact
  machine-readable status.
- FD004 training provenance remains historical and incomplete.
- Manifest-generation time is not training time.
- The new current source hash must not be described as the historical training
  source hash.
- `config_file_sha256` is raw-byte integrity and
  `config_canonical_sha256` is versioned semantic identity; neither may be
  mislabeled as a historical training hash for a post-training normalized config.
- Model binaries must not change.

### 11.6 Load-time verification

When a current manifest/metadata schema is available, verify config/model/preprocessor
hashes before deserialization in:

- FD001 serving;
- FD001 conformal/post-hoc evaluation;
- FD004 post-hoc evaluation.

Keep these error classes distinct:

1. artifact absent - friendly generation guidance;
2. legacy artifact without current integrity schema - explicit compatibility
   message or deliberately documented legacy path;
3. present artifact hash/identity mismatch - hard integrity failure.

### Gate P4

- Tracked verification passes in a clean clone without local models.
- Full verification passes locally with all four frozen artifacts.
- Tampering a temporary copy fails before deserialization.
- Missing-local and wrong-local semantics are tested.
- `--check` never writes; fixed-input/fixed-timestamp manifest generation is
  byte-for-byte deterministic, and unchanged inputs preserve their generation time.
- Tamper tests use a temporary `root` override and never touch real frozen files.
- All four baseline hashes remain unchanged.

---

## 12. Phase 5 - Repository-Wide Reference and Encoding Integrity

### 12.1 Checker

Create `scripts/check_repository_integrity.py` with independently callable:

- `check_references()`;
- `check_text_encoding()`;
- a CLI returning nonzero on violations.

Enumerate candidate files through `git ls-files`, not recursive filesystem walking.

### 12.2 Reference behavior

Recognize Markdown link targets and path-like tokens in tracked Markdown and Python
documentation/comments. Support known repository extensions such as `.md`, `.py`,
`.yaml`, `.yml`, `.toml`, `.json`, `.csv`, `.txt`, `.ipynb`, `.joblib`, `.keras`,
and `.npz`.

Resolve against repository root, the referring file's directory, and a unique
tracked-basename match. Reject missing or ambiguous tracked references with source
file and line.

Exclude URLs, anchors, documented templates/placeholders, and intentionally
gitignored generated artifacts. Any unavoidable exception must be stored in
`configs/repository_integrity.yaml` with source, target/pattern, and a nonempty
reason. Fail on unused exceptions so the list cannot become permanent clutter.

Retain a small explicit required-anchor assertion for the four restored plans and
key V2.2 evidence files.

### 12.3 Encoding behavior

Strictly decode tracked `.md`, `.py`, `.toml`, `.yaml`, `.yml`, and `.txt` files as
UTF-8. Reject U+FFFD and known mojibake sequences. Repair each affected line through
explicit reviewed replacements; do not apply repeated blind encode/decode
transformations.

Known replacements include corrupted em/en dashes, arrows, ellipses,
multiplication signs, plus/minus, squared, alpha, minus, approximately, and section
sign sequences. Inspect `CHANGELOG.md`, `tests/test_v2_2_cleanup.py`, `AUDIT_V2.md`,
the restored plans, and all other hits.

Optionally add `.editorconfig` and narrowly scoped `.gitattributes` only if doing so
does not cause a repository-wide line-ending rewrite. Avoid noisy mechanical diffs.

### Gate P5

- All four restored plans and all live references resolve.
- No unexplained missing/ambiguous reference remains.
- No invalid UTF-8, U+FFFD, or known mojibake remains.
- Every exception is narrow, used, and justified.
- `git diff --check` is clean.

---

## 13. Phase 6 - Test-Count and Documentation Truth

### README

- Keep stable test commands and tier explanations.
- Remove permanent exact current pass/skip/deselect counts.
- Add an actual CI badge only if the repository has a verifiable GitHub remote and
  workflow URL. Do not invent a badge URL.
- Link to the dated historical measurement section in the final report.

### PROJECT_SPEC

- Describe artifact-free CI, tracked-artifact falsification, local artifact suites,
  and the historical report without numeric "current" counts.

### CHANGELOG and final report

- Preserve genuine historical counts.
- Label the 2026-08-18 measurements as a snapshot at commit `23cc934`.
- State that later commits can change collection/counts and that CI is authoritative
  for current branch health.
- Do not replace failed current tests with new prose counts before fixing the tests.

### Documentation tests

Add guards preventing README/PROJECT_SPEC from reintroducing unlabeled "current: N
passed" or mutable measured-count blocks.

### Gate P6

- Stable docs contain no permanent mutable numeric current-count claim.
- Historical counts include date/commit context.
- Current status points to real CI or reproducible commands.
- All config/provenance/manifest wording matches actual fields and behavior.

---

## 14. Phase 7 - Import-Safe Streamlit Application

Refactor `app_v2.py` around:

```python
@st.cache_resource
def get_predictor():
    from rul_prediction.serving.v2_predictor import V2Predictor
    return V2Predictor()


def main():
    predictor = get_predictor()
    # all Streamlit rendering and data access


if __name__ == "__main__":
    main()
```

Keep only lightweight definitions/imports at module scope. Move data loading,
predictor construction, rendering, and predictions into `main()` or its called
functions.

In `V2Predictor.__init__`, import TensorFlow/Keras only in the neural-model branch.
The current XGBoost deployment must use joblib without initializing TensorFlow.

Add `tests/test_app_v2.py` covering:

- artifact-free subprocess import succeeds;
- `main` and `get_predictor` exist;
- import does not construct a predictor;
- import does not open models/scalers/raw data;
- import emits no `missing ScriptRunContext` or TensorFlow initialization output;
- friendly missing-artifact guidance appears only when the app actually requests a
  predictor;
- a headless Streamlit launch works when local artifacts are present and is marked
  `app_smoke` plus supplemental `needs_artifacts`.

Do not use a brittle hard timing assertion; record timing diagnostically.

### Gate P7

- Bare import is side-effect free in a clean clone.
- XGBoost serving does not import TensorFlow.
- Real headless Streamlit smoke still works locally.
- Existing UI claims, prediction fields, and missing-artifact guidance remain intact.

---

## 15. Phase 8 - Test Taxonomy and CI Structure

### 15.1 Markers

Register and use:

- `unit` - pure, synthetic, fast;
- `static_contract` - source/config/reference/encoding contracts;
- `tracked_artifacts` - committed CSV/JSON/YAML evidence;
- `integration` - multiple components;
- `app_smoke` - app import/startup behavior;
- `needs_artifacts` - supplemental requirement for gitignored raw/model/processed
  files.

Every test must have exactly one primary marker. `needs_artifacts` is supplemental.
Enforce this with a collection hook or marker-audit test.

### 15.2 Classification rules

- Tests opening `data/raw`, `data/processed`, `models`, or other gitignored generated
  files require `needs_artifacts`.
- Tests reading tracked `experiments/v2_2` evidence use `tracked_artifacts` and remain
  active in CI.
- Known missing artifacts must be represented by markers, not opportunistic runtime
  skips.
- Split the monolithic cleanup test by concern when practical, without losing
  coverage.
- Remove or replace the nested full-pytest collection test that makes one unit test
  launch a second repository-wide collection. Marker auditing should be direct and
  nonrecursive.

### 15.3 CI jobs/steps

Run these exact selectors, with clear failure attribution:

```bash
pytest -m static_contract
pytest -m unit
pytest -m tracked_artifacts
pytest -m "integration and not needs_artifacts"
pytest -m "app_smoke and not needs_artifacts"
pytest -m "not needs_artifacts"
pytest -m needs_artifacts --collect-only
```

They correspond to:

1. dependency consistency and `pip check`;
2. repository integrity/static contracts;
3. unit tests;
4. tracked-artifact falsification;
5. artifact-free integration;
6. app import smoke;
7. aggregate `pytest -m "not needs_artifacts"` guard;
8. collect `needs_artifacts` tests to prove discoverability.

The CI app-smoke tier covers artifact-free import/startup behavior. A real headless
Streamlit smoke that loads gitignored artifacts must carry both `app_smoke` and
`needs_artifacts` and runs only in the local full-artifact gate, never in clean-clone
CI.

### Gate P8

- Every collected test has exactly one primary tier.
- Every gitignored-artifact dependency is marked.
- Tracked-evidence tests execute in a clean clone.
- CI failures identify a tier.
- Aggregate artifact-free and full local suites remain green.

---

## 16. Phase 9 - Dependency and CI Reproducibility

The current `requirements-lock.txt` is a Windows Python 3.12.6 `pip freeze`, while
the V2.2 configs record Python 3.12.10 and CI currently requests floating 3.12.
Treat the file as pinned constraints unless/until a cross-platform lock is generated.

### Minimal-churn implementation

1. Validate/regenerate the constraints in a clean Python 3.12.10 environment.
2. Remove local/editable/path/VCS dependencies.
3. Add environment markers for platform-only constraints where required.
4. Update the header with role, generation command, Python version, platform, and
   install command.
5. Pin CI to Python 3.12.10 if available and consistent with project policy.
6. Select and record an exact pip installer version for cold-install evidence; CI
   must install that version rather than floating to the newest pip.
7. Install in CI with:

```bash
python -m pip install "pip==<validated-version>"
python -m pip install -r requirements.txt -c requirements-lock.txt
python -m pip install -e . --no-deps
python -m pip check
```

8. If the Windows constraints cannot resolve on Ubuntu, create
   `requirements-ci-linux-py312.txt`. Ubuntu CI must select that file explicitly;
   Windows/local reproduction continues to select `requirements-lock.txt`. Do not
   fall back to unconstrained CI merely to obtain green tests.
9. Define dependency agreement precisely: normalized `[project.dependencies]`
   runtime names/specifiers must be a subset of direct `requirements.txt` entries.
   Additional direct requirements are allowed only when classified in a documented
   ML, application, test, or notebook category. Do not require exact equality or
   introduce packaging extras in this repair.
10. Add a dependency consistency checker ensuring direct requirements are governed
    by selected constraints, the subset/category contract holds, no path
    dependencies exist, CI consumes the platform-selected constraints, and Python
    policy agrees.

Artifact manifests must distinguish historical and present verification
environments. Record `historical_constraints_path/sha256` only when supported by
historical evidence, and record the exact current
`verification_constraints_path/sha256` used by the verifier. Never describe the new
Linux constraints file as the historical training environment.

Avoid a broad packaging/extras redesign unless it is required to make the constrained
install correct. Record any sensible extras split as a separate follow-up.

### Disposable-environment loop

1. Create a new temporary virtual environment.
2. Install requirements plus constraints.
3. Record Python, pip, selected constraints path/hash, and platform, then run
   `pip check`.
4. Compare TensorFlow, NumPy, pandas, scikit-learn, XGBoost, and joblib to the
   documented V2.2 versions.
5. Run repository-integrity, unit, tracked-artifact, app-import, and artifact-free
   suites.
6. Repeat in Linux CI.

### Gate P9

- CI consumes pinned constraints and runs `pip check`.
- CI uses a documented platform-to-constraints selection rule and a recorded exact
  pip version.
- No local/editable/path dependency is embedded in the lock.
- Pyproject runtime dependencies are a subset of direct requirements; additions are
  assigned to documented ML/app/test/notebook categories.
- Python/package wording is consistent.
- A cold environment resolves and runs the artifact-free suite.

---

## 17. Phase 10 - Cross-Cutting Documentation and Implementation Report

After code and tests stabilize, the lead agent alone reconciles:

- `README.md`;
- `PROJECT_SPEC.md`;
- `CHANGELOG.md`;
- `reports/v2_2_final_report.md`;
- historical-snapshot explanations outside the byte-identical restored plan files;
- this plan's issue tracker.

Create `reports/repository_integrity_implementation_report.md` containing:

- implementation date and commits;
- baseline and final worktree states;
- restored historical files and blob evidence;
- exact changed-file inventory by workstream;
- no-retrain/no-rerun declaration;
- before/after binary hashes;
- source-hash algorithm and dirty policy;
- artifact-manifest schema and verification modes;
- dependency installation method;
- test tiers and commands;
- targeted, full, artifact-free, and clean-clone results;
- app import/headless smoke results;
- selection, metric, and conformal falsification results;
- historical limitations and remaining follow-ups.

Any numeric test result in this report must be tied to a date, OS, Python version,
command, and exact commit/worktree hash. It is a historical measurement, not a
permanent current-count claim.

---

## 18. Mandatory Convergence Loops

### Loop L1 - Red/green/refactor loop

For every issue:

1. Reproduce it with a focused failing test or command.
2. Implement the smallest coherent behavior.
3. Run the focused test.
4. Run the owning tier.
5. Add at least one adversarial case for config, provenance, manifest, or parsing
   behavior.
6. Inspect semantic diff and `git diff --check`.
7. Repeat until green without weakening coverage.

### Loop L2 - Preservation loop

After every phase:

1. Recompute the four frozen binary hashes.
2. Recompute FD001/FD004 headline metrics from saved predictions.
3. Reapply the CV selection policy.
4. Recompute conformal q values.
5. Confirm tracked result/config changes are expected.
6. Stop immediately on unexplained binary or numerical drift.

### Loop L3 - FD004 schema/propagation loop

1. Parse production YAML.
2. Mutate one behavior-driving value in a synthetic config.
3. Confirm it reaches the appropriate runtime spy or is rejected before side effects.
4. Verify training/inference artifact identity and metadata agree.
5. Repeat until every field is covered.

### Loop L4 - Artifact-integrity loop

1. Build manifests.
2. Verify tracked mode.
3. Verify full mode.
4. Tamper with a temporary artifact copy and prove failure.
5. Test absent-local and wrong-local behavior.
6. Regenerate and confirm deterministic stable content.

### Loop L5 - Provenance falsification loop

Exercise clean, staged, unstaged, untracked source, ignored cache, deletion, rename,
Unicode/path normalization, and dirty snapshot reconstruction. Do not declare the
hash repair complete until behavior matches documentation in every case.

### Loop L6 - Dependency cold-install loop

Repeat clean environment creation, constrained install, `pip check`, core-version
comparison, integrity checks, and artifact-free tests until both local and Linux CI
are green. Do not remove constraints to bypass a resolution failure.

### Loop L7 - Documentation truth loop

1. Run the reference and encoding checker.
2. Search for mutable current-count claims.
3. Compare every config/provenance/manifest claim to code and JSON fields.
4. Compare historical measurement wording to exact commit/date evidence.
5. Fix the source claim, not only the test.
6. Repeat until checker and independent doc review are clean.

### Loop L8 - Cross-agent review loop

1. Integrate workers by file ownership.
2. Have a different sub-agent review each workstream.
3. Classify findings by severity.
4. Return findings to the owning agent or lead.
5. Fix root causes.
6. Rerun focused test, owning tier, and preservation loop.
7. Repeat until two independent final reviewers report no material findings.

### Loop L9 - Final repository loop

Run in order:

1. `git status --short --branch`.
2. `git diff --check`.
3. repository reference/encoding checker.
4. FD004 config tests.
5. provenance tests.
6. artifact-manifest tests.
7. dependency consistency and `pip check`.
8. unit tier.
9. static-contract tier.
10. tracked-artifact tier.
11. artifact-free integration tier.
12. app smoke tier.
13. aggregate artifact-free suite.
14. full local suite.
15. full local artifact verification.
16. scientific preservation/falsification checks.

For any failure: classify, send to owner, fix root cause, rerun focused and owning
tier, then restart this loop from the earliest affected layer.

### Loop L10 - Real clean-clone loop

Only after the implementation is committed:

1. Clone the exact commit into a newly created temporary directory.
2. Create a fresh environment using the constrained installation path.
3. Run `pip check`.
4. Run repository integrity.
5. Run tracked-manifest verification.
6. Run the exact CI artifact-free commands.
7. Run side-effect-free `import app_v2` without models/raw data.
8. Record commit, OS, Python, commands, date, and results.
9. Clean up only the exact verified temporary directory.
10. If any failure occurs, fix in the source repository, commit, and repeat against
    the new exact commit.

---

## 19. Stop and Escalation Conditions

Stop and ask the user before:

- retraining or rerunning FD001/FD004 selection;
- changing a frozen model/preprocessor binary;
- changing headline metrics, predictions, conformal values, or selected models;
- committing model/data artifacts;
- rewriting historical commits or metadata claims;
- deleting user-owned files;
- materially expanding from repository integrity into a new modeling methodology.

If the same blocker remains after three evidence-backed attempts, record the blocker,
commands, errors, attempted alternatives, and exact user decision required. Do not
loop indefinitely and do not mark the issue done.

---

## 20. Suggested Commit Sequence

Use the repository's normal branch policy. If a new branch is needed, prefer
`codex/repository-integrity`.

Suggested commits:

1. `docs(audit): restore methodology repair and freeze plans`
2. `fix(fd004): enforce one authoritative final configuration contract`
3. `feat(repro): add deterministic source provenance and dirty-run policy`
4. `feat(artifacts): add V2.2 integrity manifests and verification`
5. `test(integrity): add reference encoding and test-tier gates`
6. `refactor(app): make Streamlit import side-effect free`
7. `ci(deps): install with validated pinned constraints`
8. `docs(audit): reconcile historical measurements and implementation evidence`

Before every commit:

- inspect `git status --short`;
- inspect every staged diff;
- confirm no model/data/session file is staged;
- run the phase's focused tests and preservation loop;
- avoid mixing unrelated changes.

Do not commit from sub-agents independently. The lead agent owns commit boundaries.

---

## 21. Final Exit Checklist

Do not declare completion until every applicable item is checked.

### Historical integrity

- [ ] `V2_1_REPAIR_PLAN.md` restored and tracked.
- [ ] `V2_2_REPAIR_PLAN.md` restored and tracked.
- [ ] `V2_2_FINAL_CLEANUP_PLAN.md` restored and tracked.
- [ ] `V2_2_FINAL_FREEZE_PLAN.md` restored and tracked.
- [ ] All four restored blob IDs equal their `23cc934:<path>` blob IDs.
- [ ] Restored plan files are byte-identical and contain no new annotations.
- [ ] No unrelated deleted/session-only plan restored.
- [ ] All live local references resolve.
- [ ] Historical records are labeled as historical snapshots.

### Scientific preservation

- [ ] No FD001 full CV rerun performed.
- [ ] No FD004 A/B/C/D rerun performed.
- [ ] No FD001/FD004 refit performed.
- [ ] Four frozen binary hashes equal Section 4 exactly.
- [ ] FD001 CV remains 40 rows, eight candidates, folds 1-5.
- [ ] FD001 selection/champions unchanged.
- [ ] FD004 variant C remains selected.
- [ ] FD001/FD004 official metrics recompute unchanged.
- [ ] Conformal scores/q values recompute unchanged.

### FD004 authority

- [ ] One shared typed config is used by writer, freeze, naming, payload, and post-hoc.
- [ ] Architecture and clustering method fail closed when unsupported.
- [ ] Candidate identity is validated.
- [ ] Window reaches sequence creation and inference explicitly.
- [ ] Units, dropout, loss, optimizer, clipnorm, learning rate, batch, seed, and
      epochs reach runtime explicitly.
- [ ] Cluster count, seed, and `n_init` reach preprocessing explicitly.
- [ ] Split paths/counts/hashes/disjointness are config-validated.
- [ ] Final fit cannot use calibration or official-label data.
- [ ] Preprocessing payload supports the accepted variants without silent omission.
- [ ] Historical `fd004_conditionC.joblib` remains byte-identical and loads only
      through the manifest-hash-gated legacy adapter.
- [ ] Raw-file and canonical-semantic config/split hashes are separately named and
      never compared across hash kinds.
- [ ] Partial/resumed variant outputs cannot corrupt the canonical config/control rows.
- [ ] Behavioral propagation and round-trip tests pass.

### Provenance and manifests

- [ ] Tracked source hash uses `git ls-files`, not recursive filesystem scanning.
- [ ] Hash encoding is versioned, length-delimited, normalized, and fail-closed.
- [ ] Ignored caches/install metadata do not change the source hash.
- [ ] Staged/unstaged/untracked/deleted/renamed semantics are tested.
- [ ] Dirty execution is rejected or durably snapshotted with a reason.
- [ ] Unrelated non-execution dirtiness is recorded but does not require a source
      bundle; dirty snapshots pass path/size/sensitive-data safeguards and are never
      auto-staged or committed.
- [ ] Future freeze provenance is captured before training.
- [ ] Historical metadata remains unchanged and honest.
- [ ] FD001 and FD004 checked-in manifests exist.
- [ ] FD004 canonical official prediction file exists and matches its report mirror.
- [ ] Tracked and full manifest verification pass.
- [ ] Manifest `--check` is read-only and fixed-input/fixed-time output is
      byte-for-byte deterministic.
- [ ] Manifest tamper tests use a temporary root override, never real frozen files.
- [ ] Tamper/missing/wrong-local tests pass.
- [ ] Serving/evaluation verify hashes before loading when schema is available.

### Repository integrity and documentation

- [ ] Repository-wide reference checker passes.
- [ ] Encoding checker passes.
- [ ] No U+FFFD or known mojibake remains.
- [ ] Exception allowlist is empty or fully used and justified.
- [ ] README/PROJECT_SPEC contain no mutable current numeric test counts.
- [ ] Historical measurements name date and commit.
- [ ] Current health points to actual CI or commands.
- [ ] Provenance/config/manifest documentation matches code and schemas.

### Application, tests, dependencies, and CI

- [ ] `import app_v2` is artifact-free and side-effect free.
- [ ] XGBoost serving does not initialize TensorFlow.
- [ ] Headless Streamlit smoke passes with local artifacts.
- [ ] Every test has exactly one primary tier.
- [ ] Every gitignored-artifact test carries `needs_artifacts`.
- [ ] Tracked-artifact falsification runs in CI.
- [ ] Exact CI selectors in Section 15.3 run; artifact-backed app smoke is excluded
      from clean-clone CI and runs in the local full gate.
- [ ] CI consumes pinned constraints.
- [ ] CI records exact pip and selected platform constraints path/hash.
- [ ] Dependency subset/category contract passes without requiring pyproject and
      requirements exact equality.
- [ ] `pip check` passes in a disposable environment.
- [ ] Python/package-version policy is consistent.
- [ ] Unit tier passes.
- [ ] Static-contract tier passes.
- [ ] Tracked-artifact tier passes.
- [ ] Artifact-free integration tier passes.
- [ ] App-smoke tier passes.
- [ ] Aggregate artifact-free suite passes.
- [ ] Full local suite passes.
- [ ] Real clean-clone constrained install and CI-equivalent suite pass on the exact
      final commit.

### Final audit and handoff

- [ ] `git diff --check` passes.
- [ ] User's `session-ses_feb9.md` remains untouched/uncommitted.
- [ ] Independent provenance/adversarial review is clean.
- [ ] Independent repository/CI/documentation review is clean.
- [ ] Implementation report exists with exact commands and evidence.
- [ ] `reports/repository_integrity_preservation_ledger.json` validates and its
      rendered report summary matches the JSON authority.
- [ ] Issue tracker RI-01 through RI-14 contains evidence-backed final statuses.
- [ ] Working tree is clean except explicitly identified user-owned files.
- [ ] All intended changes are committed if commit authorization was granted.

---

## 22. Required Final Handoff Format

The implementing agent's final response must include:

1. Overall result: complete, partially complete, or blocked.
2. Commits and exact verified commit SHA.
3. Changed files grouped by workstream.
4. Sub-agents used, their ownership, and review outcomes.
5. No-rerun/no-retrain statement.
6. Four frozen binary hashes before and after.
7. Scientific falsification summary.
8. Test results by tier, aggregate local, artifact-free, and clean clone.
9. Manifest verification results in tracked and full modes.
10. Dependency/Python environment evidence and `pip check` result.
11. App import and headless smoke results.
12. Remaining limitations, skipped gates, or user decisions required.
13. Link to `reports/repository_integrity_implementation_report.md`.

The final response must not say `GOAL_COMPLETE`, "all fixed", or equivalent unless
every applicable checklist item above has passed with current evidence.
