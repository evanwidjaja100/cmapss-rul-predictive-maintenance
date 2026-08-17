# C-MAPSS Methodology V2.2 — Final Cleanup, Auditability & CV-Readiness Pass

## Master Instruction

You are taking over the existing repository:

```text
cmapss-rul-predictive-maintenance
```

The repository already contains **Methodology V2.2**.

V2.2 fixed the major scientific problems from earlier iterations:

- calibration engines no longer control the final FD001 fit;
- outer CV evaluation engines are separated from inner early stopping;
- nested engine-level CV exists;
- the intended 8 candidates × 5 folds = 40-run matrix was completed;
- the deployment model was selected using a pre-specified NASA-risk-first policy;
- FD001 conformal calibration uses engine-level scores;
- FD004 condition-aware modeling was rerun with clean validation separation;
- lifetime vs observed-history semantics were corrected;
- Streamlit no longer uses false OOD language;
- V2.2 protocol tests were added;
- official FD001/FD004 evaluations are correctly labeled post-hoc.

**Do not rebuild the project.**

**Do not start V2.3.**

**Do not run another full 40-fold model search unless the existing V2.2 experiment artifacts are actually missing or corrupted and cannot be recovered.**

This session is a **final cleanup / reproducibility / auditability pass**.

Your job is to fix the remaining repository inconsistencies, preserve the valid V2.2 results, rerun only the lightweight computations required by the fixes, and continue auditing in a loop until the public repository is genuinely CV-ready.

---

# 1. Operating Mode — Autonomous Repair Loop

You are authorized to work autonomously.

Use this loop:

```text
AUDIT
  ↓
DEFINE / UPDATE GOALS
  ↓
IMPLEMENT
  ↓
RUN TARGETED TESTS
  ↓
RUN ONLY REQUIRED LIGHTWEIGHT RECOMPUTATIONS
  ↓
FALSIFY RESULTS
  ↓
RE-AUDIT REPOSITORY
  ↓
IF ANY ISSUE REMAINS:
    LOOP AGAIN
ELSE:
    FINAL QA
    COMMIT
    GOAL_COMPLETE
```

Do **not** stop after one implementation pass.

Do **not** ask me whether you should continue fixing an issue that clearly belongs to this cleanup scope.

Do **not** declare completion merely because `pytest` is green.

Completion requires all of the following to agree:

- code;
- configs;
- experiment artifacts;
- serving behavior;
- documentation;
- Git status;
- clean-checkout behavior.

---

# 2. Environment Rule

Use only the project-local environment:

```text
.venv/
```

Windows interpreter:

```text
.venv\Scripts\python.exe
```

Verify first:

```bash
.venv\Scripts\python.exe -c "import sys; print(sys.executable); print(sys.prefix)"
```

Both paths must resolve inside this repository's `.venv`.

Never install dependencies globally.

Never use global `pip` after `.venv` exists.

---

# 3. Preserve History

Do **not**:

- delete V1/V2/V2.1 history;
- rewrite Git history;
- remove negative FD004 results;
- hide superseded results;
- overwrite old experiment namespaces.

Preserve historical work.

Use labels such as:

```text
SUPERSEDED BY METHODOLOGY V2.2
```

where appropriate.

V2.2 remains the current methodology.

---

# 4. Create a Cleanup Plan First

Create:

```text
V2_2_FINAL_CLEANUP_PLAN.md
```

It must contain the following columns:

```text
ID
Issue
Severity
Current evidence
Impact
Files affected
Planned correction
Required tests
Required recomputation
Status
Final evidence
```

Allowed statuses:

```text
OPEN
IN_PROGRESS
DONE
BLOCKED
```

Translate every issue below into concrete goals.

If you discover another real issue during the audit:

```text
ADD IT TO THE PLAN.
```

Do not silently ignore newly discovered defects.

A goal may only be marked `DONE` after the implementation, tests, required recomputation, artifact verification, and documentation verification are complete.

---

# 5. Absolute Scope Rule — Do Not Re-run Full CV Unnecessarily

The previous V2.2 session completed the expensive:

```text
8 candidates × 5 outer folds = 40 candidate-fold runs
```

Before doing anything expensive:

1. Search the current working tree for the existing V2.2 experiment directory.
2. Check whether the local machine still contains:

   ```text
   experiments/v2_2/
   ```

3. Verify expected V2.2 files and hashes.
4. Recover/commit those files if they exist locally but were ignored by Git.

**Do not rerun the 40-fold experiment merely because the ZIP/GitHub repo omitted `experiments/v2_2/`.**

Only rerun the expensive CV if:

- the files truly no longer exist locally;
- they cannot be recovered from the previous working tree/session;
- and there is no trustworthy way to reconstruct/validate the final summaries.

If that happens, mark it as a major recovery event in the cleanup plan before rerunning anything.

---

# 6. Known Issue 1 — `experiments/v2_2/` Is Missing From the Public Repository

The final repository currently contains V2.2 reports/configs/scripts, but the primary V2.2 audit trail is missing from the public repo.

Expected directory:

```text
experiments/v2_2/
```

Expected artifacts include at least:

```text
fd001_outer_fold_results.csv
fd001_cv_summary.csv
fd001_outer_predictions.csv
fd001_best_epochs.csv
fd001_outer_metadata.jsonl
fd001_outer_split_manifest.json
selection_decision.json

fd001_final_fit_metadata.json
fd001_final_metrics.json
fd001_official_predictions.csv

fd001_conformal_engine_scores.csv
fd001_conformal_quantiles.csv
conformal_calibration.json

fd004_variant_results.csv
fd004_variant_predictions.csv
fd004_final_model_metrics.json
```

and any other V2.2 JSON/CSV metadata created by the implementation.

The likely cause is `.gitignore`.

Historically Git allowed:

```text
!experiments/splits/
!experiments/v2_1/
```

but not:

```text
!experiments/v2_2/
```

## 6.1 Required Fix — Recover the V2.2 Experiment Directory

First determine whether:

```text
experiments/v2_2/
```

still exists locally.

If it exists:

**Do not recompute the full CV.**

Instead:

1. inspect every file;
2. verify expected row counts;
3. verify hashes where applicable;
4. verify headline metrics against those files;
5. make the directory trackable;
6. commit the appropriate lightweight CSV/JSON audit artifacts.

Update `.gitignore` appropriately, for example:

```gitignore
experiments/*
!experiments/splits/
!experiments/v2_1/
!experiments/v2_2/
```

or use a more precise pattern.

Continue ignoring:

```text
models/
data/raw/
large generated binary artifacts
```

unless there is a specific reason not to.

## 6.2 Required Fix — Experiment Completeness Check

Programmatically verify:

```text
FD001 outer fold rows = 40
```

and every candidate has exactly folds:

```text
1, 2, 3, 4, 5
```

Candidates:

```text
gru_w45_huber
gru_w60_huber
lstm_w45_huber
lstm_w60_huber
rf_w60
rf_w90
xgb_w60_d6
xgb_w90_d6
```

Required:

```text
8 × 5 = 40
```

Do not trust the report.

Read the CSV.

Fail loudly if incomplete.

## 6.3 Required Fix — Metric Falsification From Recovered Files

Recompute:

```text
candidate CV summaries
NASA mean per engine
RMSE mean/std
MAE mean/std
R² mean/std
signed bias mean/std
```

from:

```text
fd001_outer_fold_results.csv
```

and, where possible, from the prediction CSV itself.

Verify the selected candidate still matches the declared V2.2 policy.

---

# 7. Known Issue 2 — V2.2 Runs Were Executed From a Dirty / Uncommitted Worktree

The previous V2.2 experiments were run before the new V2.2 files were committed.

Therefore per-run metadata that stores:

```text
git_commit
```

may point to the previous repository HEAD rather than exactly representing the source tree used for the experiment.

Do not pretend otherwise.

## 7.1 Required Fix — Preserve but Clarify Provenance

Do not modify historical metadata to fabricate a different commit.

Instead document:

- the Git HEAD that existed during the run;
- that the V2.2 implementation files were uncommitted at experiment time;
- that the recorded development session provides chronological evidence;
- that experiment CSVs/configs/reports were later preserved.

Use precise wording.

Do **not** claim:

```text
this Git commit exactly reproduces every V2.2 experiment run
```

unless that is literally true.

## 7.2 Required Fix — Change “Pre-registered” Wording

The selection policy was specified before inspecting the final V2.2 comparison results in the recorded development session.

However, because the plan itself was not committed before the run, Git cannot independently prove a formal pre-registration.

Replace public wording such as:

```text
pre-registered selection policy
```

with:

```text
pre-specified selection policy
```

or:

```text
selection policy specified before the V2.2 model comparison in the recorded development session
```

Historical internal notes may mention the original term, but current README/report claims must be precise.

Do not rerun CV solely to create Git proof.

## 7.3 Required Fix — Improve Future Metadata

For future runs, extend metadata to include:

```text
git_commit
git_is_dirty
git_diff_hash
source_tree_hash or equivalent
timestamp
config hash
manifest hashes
```

Implement a helper if appropriate.

The current V2.2 run should honestly show that it originated from a dirty working tree if that is the truth.

Do not rewrite past facts.

---

# 8. Critical Cleanup Issue 3 — FD001 YAML Is Not the True Source of Truth

Current:

```text
configs/final_model_v2_2_fd001.yaml
```

contains fields that do not accurately describe the deployed XGBoost model.

For example, the deployed XGBoost factory historically uses approximately:

```text
learning_rate = 0.05
max_depth = 6
subsample = 0.8
colsample_bytree = 0.8
random_state = 42
```

while the YAML may still contain deep-model fields such as:

```text
learning_rate = 0.001
Adam
dropout
architecture sizes
```

These are irrelevant or wrong for XGBoost.

The freeze script also still hardcodes some model values.

This must be corrected.

## 8.1 Required Fix — Model-Type-Specific YAML

Rewrite:

```text
configs/final_model_v2_2_fd001.yaml
```

so it accurately represents the deployed XGBoost.

A possible structure is:

```yaml
methodology_version: "2.2"
dataset: FD001
target: raw RUL regression

model:
  candidate_name: xgb_w90_d6
  architecture: xgboost
  window: 90
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  n_estimators: 92
  random_state: 42
  early_stopping_rounds: null
```

**Do not blindly use these values.**

Verify the actual existing XGBoost factory and V2.2 training code first.

Write the values actually used.

Remove irrelevant fields such as:

```text
Adam optimizer
dropout
GRU layer sizes
```

from the XGBoost config.

## 8.2 Required Fix — Freeze Script Must Consume the YAML

`scripts/run_v2_2_freeze.py` must not contain hidden duplicate hyperparameters such as:

```python
max_depth = 6
```

if `max_depth` exists in YAML.

Instead use values resolved from the config.

For example:

```python
model.set_params(
    max_depth=cfg["model"]["max_depth"],
    learning_rate=cfg["model"]["learning_rate"],
    subsample=cfg["model"]["subsample"],
    colsample_bytree=cfg["model"]["colsample_bytree"],
    n_estimators=cfg["model"]["n_estimators"],
    random_state=cfg["model"]["random_state"],
)
```

Adapt this to the actual implementation.

The YAML must determine the deployed model.

## 8.3 Required Config Test

Add a test that creates a temporary test YAML with modified harmless values and verifies that the freeze/config parser resolves the YAML values rather than hidden constants.

For XGBoost verify at least:

```text
window
max_depth
learning_rate
subsample
colsample_bytree
n_estimators
seed/random_state
```

are config-driven.

---

# 9. Known Issue 4 — FD004 YAML Is Also Not Fully Authoritative

Repeat the same cleanup for:

```text
configs/final_model_v2_2_fd004.yaml
scripts/run_v2_2_fd004_freeze.py
```

Current FD004 values such as:

```text
window = 45
Huber loss
KMeans/regime count
selected variant C
GRU architecture
dropout
learning rate
batch size
fixed epoch count
```

must come from YAML.

Do not leave duplicate training constants hidden in the freeze script.

## 9.1 Required FD004 Config Structure

The exact structure may differ, but the YAML must explicitly encode:

```text
dataset: FD004
target: raw RUL

model:
  architecture
  window
  recurrent units
  dropout
  loss
  learning rate
  batch size
  fixed epochs
  seed

condition_preprocessing:
  variant
  clustering method
  number of clusters/regimes
  operating settings used
  scaling method
  seed

training:
  final engine count
  reserved engine count
  fixed epoch rule
```

Verify actual values first.

Do not copy assumptions from this instruction blindly.

## 9.2 Lightweight Verification After Config Cleanup

Because the deployed FD001 model is XGBoost, re-freezing FD001 should be fast.

Run:

```text
selection/config validation
final FD001 freeze
post-hoc metric verification
```

Verify new predictions match expected results within deterministic tolerance.

For FD004, re-freezing the final selected model may take longer but is still far cheaper than rerunning the full A/B/C/D search.

Do **not** rerun the full FD004 variant search unless config cleanup reveals the existing outputs cannot be trusted.

---

# 10. Critical Cleanup Issue 5 — Serving Short-History Risk Flag Is Not Defensible

Current serving uses an empirical short-history threshold.

The previous error-analysis implementation appears to derive the threshold incorrectly because history-bin lower bounds may be strings.

Example lexical behavior:

```text
"90" > "128"
```

which can produce the wrong numeric maximum.

This must be investigated.

## 10.1 Required Fix — Confirm the String/Non-Numeric Bug

Inspect:

```text
scripts/analyze_v2_2_errors.py
```

and any code deriving:

```text
RISK_OBSERVED_CYCLES
```

Verify whether bucket values are being compared as strings.

Add a regression test.

Example numeric candidates:

```text
45
90
128
```

must produce numeric behavior, never lexicographic behavior.

## 10.2 Required Fix — Remove Post-Hoc Test-Driven Risk Threshold From Deployment

More importantly:

the official FD001 test labels were used during post-hoc error analysis.

Therefore a threshold derived from that official-test error analysis should **not** drive prospective Streamlit warning behavior.

Remove the post-hoc risk threshold from serving unless it is re-derived strictly from development CV data.

Preferred final serving behavior:

```python
history_is_padded = observed_cycles < model_window
n_padded_timesteps = max(model_window - observed_cycles, 0)
```

Display objective information such as:

```text
63 cycles are observed while the model window is 90; 27 leading timesteps were padded.
```

This is fully defensible.

Do not call this OOD.

Do not call it a failure-risk score.

## 10.3 Optional Development-Derived Risk Flag

If you genuinely want a separate empirical risk warning:

derive the threshold using only:

```text
V2.2 OUTER CV PREDICTIONS
```

Never official FD001 test labels.

Predefine the rule before calculating the threshold.

For example:

```text
find history-length bins whose macro-engine error or NASA contribution is materially worse than the development baseline
```

But this is optional.

Simpler is better:

**remove the empirical deployment risk flag.**

Keep only padding/history facts.

## 10.4 Required Serving Tests

Test:

```text
observed_cycles < window
→ history_is_padded = True
→ n_padded_timesteps correct
```

and:

```text
observed_cycles >= window
→ history_is_padded = False
```

Ensure no current serving field is called:

```text
OOD
out_of_distribution
lifetime_risk
official-test-derived risk
```

---

# 11. Cleanup Issue 6 — Sensitivity Analysis Uses Future Information

Current V2.2 sensitivity analysis must be audited carefully.

The sensor replacement baseline must only use data available up to the prediction cutoff.

For a cutoff at cycle `t`:

Allowed:

```text
cycle <= t
```

Forbidden:

```text
sensor values from cycle > t
```

## 11.1 Required Fix — Prefix-Only Sensor Baseline

For each engine and cutoff:

```python
observed_history = engine[engine["cycle"] <= cutoff]
sensor_replacement_value = observed_history[sensor].mean()
```

Do not use the mean over the engine's full run-to-failure history.

Do not use future sensor rows.

## 11.2 Required Fix — RMSE Delta Alignment

Audit how:

```text
baseline predictions
occluded predictions
true RUL
```

are aligned.

Do not depend on accidental `groupby` ordering.

Build a dataframe keyed by:

```text
engine_id
cutoff_cycle
true_rul
baseline_prediction
occluded_prediction
```

Then calculate:

```python
baseline_error = baseline_prediction - true_rul
occluded_error = occluded_prediction - true_rul
```

and derive RMSE values from exactly aligned rows.

## 11.3 Required Fix — Rerun Sensitivity Only

Do not rerun model training.

Rerun:

```text
scripts/explain_v2_2_sensitivity.py
```

using the existing/final V2.2 model.

Regenerate:

```text
reports/v2_2_sensitivity.md
reports/tables/v2_2_sensor_sensitivity.csv
reports/tables/v2_2_temporal_sensitivity.csv
```

Do not preserve old rankings as current unless the corrected calculation reproduces them.

Report whatever the corrected analysis shows.

## 11.4 Required Sensitivity Tests

Test that for cutoff cycle 50:

```text
replacement baseline never reads cycle 51+
```

Create a synthetic engine where future rows contain extreme values and verify that the prefix-only baseline is unaffected.

Also test row alignment by deliberately scrambling engine/cutoff ordering.

Metric output must remain identical.

---

# 12. Cleanup Issue 7 — Model-Selection Bias Tie-Break Has a Bug

Declared rule:

```text
prefer smaller ABSOLUTE signed bias
```

Implementation may currently sort directly by:

```text
signed_bias_mean
```

instead of:

```text
abs(signed_bias_mean)
```

This is wrong.

Example:

```text
bias = -20
```

must not beat:

```text
bias = +1
```

when the rule is smaller absolute bias.

## 12.1 Required Fix

Implement:

```python
abs_signed_bias = abs(signed_bias_mean)
```

and use that in the tie-break.

Add tests:

```text
candidate A bias = -20
candidate B bias = +1
```

Candidate B must win the bias tie-break.

Also verify that the actual V2.2 deployment candidate remains unchanged.

The previous actual selection should not have depended on this tie-break because the NASA difference exceeded the pooled-SE guardrail.

Programmatically verify that.

---

# 13. Cleanup Issue 8 — Conformal Wording Still Overstates Historical Cleanliness

Mechanically, V2.2 conformal is now:

```text
15 calibration engines
×
five fixed lifecycle checkpoints
→ one max absolute error per engine
→ 15 scores
```

That is good.

But those calibration engines had already been inspected in earlier project iterations.

Therefore do not claim:

```text
pristine
never seen before
first label contact in project history
one-shot untouched conformal calibration
```

The V2.2 model fit did not use them.

That is the important claim.

## 13.1 Required Conformal Wording

Use wording similar to:

> The V2.2 interval uses engine-level split-conformal mechanics on 15 engines held out from V2.2 fitting and model selection. These engines were inspected during earlier project iterations, so the resulting interval should be interpreted as an empirically calibrated uncertainty interval rather than a pristine one-shot external conformal guarantee.

Also retain:

> Application to arbitrary uploaded trajectories is an engineering extrapolation beyond the fixed checkpoint calibration scheme.

Update:

```text
README
PROJECT_SPEC
reports
Streamlit
config comments
```

where relevant.

Do not change the actual `q` merely for wording cleanup.

---

# 14. Cleanup Issue 9 — Clean-Checkout Test Counts Are Overstated

The previous agent reported:

```text
134 full tests
130 artifact-free
```

but that "artifact-free" run was executed in the existing development environment where raw/generated artifacts were still present.

A genuine public ZIP/clean-clone audit may instead produce skipped tests because:

```text
data/raw
models
generated artifacts
```

are absent.

## 14.1 Required Fix — Real Clean-Checkout Simulation

Create a temporary clean source tree.

Do not use the existing artifact-rich working tree.

The simulation must omit or isolate:

```text
data/raw
data/processed
models
generated experiment artifacts not tracked by Git
local caches
```

Then run the same CI test command.

Record:

```text
passed
skipped
deselected
failed
```

This is the number the README should call:

```text
clean checkout / CI behavior
```

## 14.2 Required Fix — Test Documentation

Clearly distinguish:

```text
Full local artifact-rich suite
```

from:

```text
Clean public checkout
```

Example format:

```text
Local development tree:
134 passed

Clean source checkout:
X passed
Y skipped
Z deselected
```

Use the actual values from your rerun.

Do not copy example counts.

## 14.3 Required Fix — CI Should Match the Documented Command

Inspect:

```text
.github/workflows/ci.yml
```

Ensure GitHub Actions runs the same intended artifact-free suite.

Dataset-dependent tests must either:

- skip cleanly; or
- be marked `needs_artifacts`.

Do not require NASA data in CI.

---

# 15. Cleanup Issue 10 — Streamlit Depends on an Untracked Conformal File

Current predictor may expect:

```text
experiments/v2_2/fd001_conformal_quantiles.csv
```

If `experiments/v2_2` is missing from the public repo, Streamlit cannot load its uncertainty value.

Even after committing `experiments/v2_2`, deployment configuration should not depend on a large experiment folder unnecessarily.

## 15.1 Required Fix — Track Deployment Uncertainty Config

Preferred:

place final deployment uncertainty values in the final model config or a small tracked deployment config.

For example:

```text
configs/deployment_v2_2_fd001.yaml
```

or inside:

```text
configs/final_model_v2_2_fd001.yaml
```

Possible structure:

```yaml
uncertainty:
  method: engine_cluster_conformal
  alpha: 0.10
  q: <actual value>
  calibration_engine_count: 15
  checkpoint_fractions:
    - 0.25
    - 0.45
    - 0.65
    - 0.80
    - 0.95
  interpretation: empirical_v2_2_calibration
```

The Streamlit app should be able to load its final `q` from a tracked configuration.

The experiment CSV remains the audit source.

## 15.2 Required Fix — Missing Report Reference

Audit all Streamlit links.

The app may currently refer to:

```text
reports/v2_2_methodology.md
```

while the actual report may be:

```text
reports/v2_2_final_report.md
```

Fix all broken references.

Add a small test or static path validation.

---

# 16. Cleanup Issue 11 — Documentation / Source-of-Truth Sweep

After all technical changes settle, audit:

```text
README.md
PROJECT_SPEC.md
CHANGELOG.md
V2_2_REPAIR_PLAN.md
V2_2_FINAL_CLEANUP_PLAN.md
reports/v2_2_*.md
configs/final_model_v2_2_*.yaml
app_v2.py
serving code
```

Search for:

```text
pre-registered
pristine
never touched
first time calibration labels
130 artifact-free
134 tests
source of truth
OOD
out-of-distribution
RISK_OBSERVED_CYCLES
short-history risk
SHAP
models/v2_1
q = 70.34
experiments/v2_2
v2_2_methodology.md
```

For every occurrence:

verify it is correct.

Do not blindly delete historical text.

Historical text may remain if clearly marked historical/superseded.

---

# 17. Current Model Results Must Not Be Changed Without Evidence

The expected current V2.2 public results are approximately:

```text
FD001 deployment:
xgb_w90_d6

CV NASA/engine:
~368.02

accuracy champion:
lstm_w60_huber
RMSE ~26.19

post-hoc FD001:
RMSE ~26.25
NASA ~60,963.8

FD004 selected variant:
C

post-hoc FD004:
RMSE ~33.66
```

Do not manually force the repository back to these numbers.

Instead:

**recompute from the actual recovered artifacts.**

If the files support different values:

```text
REPORT THE DIFFERENCE.
```

Never edit metrics by hand.

---

# 18. Targeted Recomputation Allowed

You may rerun:

- config parsers;
- final XGBoost freeze;
- post-hoc FD001 inference;
- conformal inference;
- sensitivity analysis;
- error analysis;
- clean-checkout test suite;
- Streamlit smoke tests;
- FD004 final freeze if necessary;
- metric falsification.

You should **not** rerun:

- the full FD001 40-run CV;
- the full FD004 A/B/C/D experiment;

unless artifact recovery/integrity genuinely fails.

---

# 19. Experiment Artifact Integrity

For recovered `experiments/v2_2` files, verify at minimum:

```text
CV row count
candidate/fold uniqueness
manifest hashes
selection decision
best iterations
final duration
official prediction counts
conformal score count
FD004 result rows
```

Expected structural invariants:

## FD001

```text
40 candidate-fold rows
5 folds per candidate
15 calibration engine scores
100 official FD001 predictions
```

## FD004

```text
248 official predictions
```

Do not infer these from reports.

Count actual rows.

---

# 20. Final Git Practice

This session must end with the cleanup committed unless a genuine Git blocker exists.

Before commit:

```bash
git status --short
```

Ensure intended V2.2 files are tracked.

Do not leave the following untracked:

```text
important configs
reports
tests
experiment CSV/JSON audit artifacts
scripts
```

Do not commit:

```text
.venv
raw NASA data
large local caches
session-state directories
large trained models unless repository policy explicitly allows them
```

Commit logically.

Suggested commits:

```text
fix(v2.2): make configs authoritative and clean serving semantics

analysis(v2.2): correct prefix-only sensitivity attribution

repro(v2.2): track experiment audit artifacts and clean-checkout tests

docs(v2.2): finalize reproducibility and uncertainty disclosures
```

You may combine logically if appropriate.

---

# 21. Required Test Additions

At minimum add or update tests for:

1. XGBoost config fully controls final model parameters.
2. FD004 config controls final freeze parameters.
3. Absolute-bias tie-break.
4. Numeric history-threshold behavior if threshold code remains.
5. Serving contains no test-derived empirical risk threshold.
6. Prefix-only sensitivity baseline.
7. Sensitivity row alignment.
8. Deployment config contains uncertainty `q`.
9. Streamlit referenced report paths exist.
10. `experiments/v2_2` structural completeness when artifacts are present.
11. Clean checkout skips artifact-dependent tests correctly.

---

# 22. Looping Protocol

After every logical repair group:

1. update `V2_2_FINAL_CLEANUP_PLAN.md`;
2. run targeted unit tests;
3. run lightweight recomputation if required;
4. inspect generated artifacts;
5. compare generated values with configs;
6. grep for stale claims;
7. try to falsify your own conclusion;
8. mark `DONE` only with evidence.

If something fails:

```text
LOOP.
```

Do not wait for me.

---

# 23. Self-Audit Round 1 — Reproducibility

Before completion answer, with evidence:

- Is `experiments/v2_2` now publicly tracked?
- Can all 40 CV rows be inspected?
- Does every candidate have 5 folds?
- Do configs match the actual model hyperparameters?
- Do freeze scripts consume config values rather than duplicate them?
- Can serving load `q` without depending on an ignored file?
- Do tracked prediction CSVs reproduce headline metrics?
- Are current experiment metadata limitations disclosed?
- Is the current source tree committed?

Any `NO`:

```text
FIX IT.
```

---

# 24. Self-Audit Round 2 — Scientific Claims

Try to reject the project.

Ask:

- Is official test information affecting live serving decisions?
- Does sensitivity use future observations?
- Is the conformal guarantee overstated?
- Is the selection policy implemented exactly as written?
- Is absolute bias really absolute?
- Are CV results still being described as externally untouched?
- Are FD001/FD004 official metrics correctly post-hoc?
- Are any historical V2/V2.1 numbers presented as current?

Any credible fixable criticism:

```text
FIX IT.
```

---

# 25. Self-Audit Round 3 — Clean Clone

Create or simulate a fresh source checkout.

Verify:

- installation instructions;
- artifact-free tests;
- skipped integration tests;
- package imports;
- config parsing;
- app import behavior when models are absent;
- error message explaining how to generate/download artifacts;
- tracked reports/configs/experiments all exist.

Record actual results.

---

# 26. Final Exit Checklist

Do not emit `GOAL_COMPLETE` until all applicable items pass.

```text
[ ] V2_2_FINAL_CLEANUP_PLAN.md exists

[ ] experiments/v2_2 recovered or explicitly documented unrecoverable

[ ] experiments/v2_2 trackable in .gitignore

[ ] 40/40 FD001 CV matrix present in public repo

[ ] every candidate has folds 1–5

[ ] selection_decision.json tracked

[ ] best-epoch / best-iteration evidence tracked

[ ] conformal engine-score audit file tracked

[ ] final-fit metadata tracked

[ ] V2.2 metadata provenance honestly describes dirty historical run

[ ] public wording says pre-specified rather than unverifiable pre-registered

[ ] FD001 YAML matches actual XGBoost hyperparameters

[ ] irrelevant deep-model parameters removed from XGBoost config

[ ] FD001 freeze fully config-driven

[ ] FD004 YAML matches actual final model/preprocessing

[ ] FD004 freeze fully config-driven

[ ] bias tie-break uses abs(bias)

[ ] deployment candidate still reproduced by policy

[ ] post-hoc test-derived risk threshold removed from serving OR re-derived development-only

[ ] numeric threshold bug fixed if threshold machinery remains

[ ] serving exposes objective padding/history fields

[ ] sensitivity baseline uses prefix-only observed history

[ ] sensitivity row alignment is deterministic

[ ] corrected sensitivity report regenerated

[ ] conformal historical wording corrected

[ ] no pristine/never-seen calibration claim remains

[ ] uncertainty q available from tracked deployment/final config

[ ] Streamlit references only existing files

[ ] clean-checkout test behavior measured from a real clean source tree

[ ] README distinguishes local artifact-rich vs clean checkout tests

[ ] CI command matches documentation

[ ] current test counts accurate

[ ] FD001 saved predictions recompute headline metrics

[ ] FD004 saved predictions recompute headline metrics

[ ] current configs/reports agree

[ ] stale-claim grep passes

[ ] full local tests pass

[ ] clean-checkout/CI tests pass or intentionally skip

[ ] app import/smoke passes

[ ] Git working tree clean except intentionally ignored local/session files

[ ] all cleanup changes committed
```

---

# 27. Final Completion Report

Only when the exit checklist passes, provide one final report with the following sections.

## 27.1 Cleanup Summary

Explain what remained wrong and what changed.

## 27.2 Experiment Artifact Recovery

State whether `experiments/v2_2` was recovered without rerunning CV.

Report:

```text
40/40 status
files committed
hash/integrity checks
```

## 27.3 Configuration Truthfulness

Show final FD001 and FD004 model hyperparameters and confirm freeze scripts consume YAML.

## 27.4 Serving Cleanup

Explain:

```text
padding behavior
removal/replacement of test-derived risk threshold
tracked conformal q source
```

## 27.5 Sensitivity Correction

Explain:

```text
old future-information defect
prefix-only replacement method
corrected sensor ranking
```

## 27.6 Selection Policy

Confirm:

```text
NASA champion
accuracy champion
deployment selection
absolute-bias bug fix
whether selected model changed
```

## 27.7 Conformal Interpretation

State:

```text
mechanical calibration method
historical calibration-set caveat
formal limitations
post-hoc empirical coverage
```

## 27.8 Reproducibility

Explain:

```text
dirty historical V2.2 run provenance
new metadata behavior
Git commits
experiment artifact tracking
```

## 27.9 Testing

Report separately:

```text
full local artifact-rich suite
true clean-checkout suite
skips/deselections
CI command
app smoke
```

## 27.10 Metric Falsification

Report independently recomputed:

```text
FD001 RMSE / MAE / R² / NASA
FD004 RMSE / MAE / R² / NASA
```

## 27.11 Remaining Limitations

Be explicit.

## 27.12 CV Readiness

State exactly one:

```text
CV-READY
```

or:

```text
NOT CV-READY
```

with evidence.

---

# 28. Completion Signal

Only after the complete exit checklist passes, end with:

```text
GOAL_COMPLETE: V2.2 final cleanup is complete and the repository is CV-ready.
```

Then provide concise evidence:

```text
- experiments/v2_2 publicly tracked
- 40/40 matrix verified
- YAMLs authoritative
- no test-derived serving threshold
- sensitivity corrected
- bias tie-break corrected
- conformal wording corrected
- clean-checkout behavior verified
- metrics falsified from saved predictions
- tests green
- Git clean and committed
```

If a genuine blocker remains, end instead with:

```text
GOAL_BLOCKED:
```

and provide exact technical evidence.

---

# 29. Begin Now

Immediately do the following:

1. Verify `.venv`.
2. Inspect Git status and `.gitignore`.
3. Locate the local `experiments/v2_2` directory.
4. Do **not** rerun full CV yet.
5. Create `V2_2_FINAL_CLEANUP_PLAN.md`.
6. Verify the 40-row CV matrix from recovered artifacts.
7. Compare final YAMLs with the actual model factories and freeze scripts.
8. Audit serving risk-threshold derivation.
9. Audit V2.2 sensitivity for future-information leakage.
10. Audit selection tie-break implementation.
11. Audit conformal wording and historical calibration provenance.
12. Perform a true clean-checkout CI simulation.
13. Implement fixes.
14. Loop through tests, recomputation, falsification and repository audit.
15. Commit all required cleanup/audit artifacts.
16. Stop only at `GOAL_COMPLETE` or a genuine `GOAL_BLOCKED` condition.
