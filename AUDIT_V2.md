# AUDIT_V2.md — Baseline Audit for Methodology V2

Date: 2026-08-15
Branch: `main` (`13ad6c3 docs: finalize README and portfolio handoff materials`)
Status: audit + preservation only. No ML methodology changes made in this phase.

## 1. Repository state

- Clean working tree, 15 commits, single branch `main`.
- `.venv` present; interpreter resolves to `D:\Desktop\test\cmapss-rul-predictive-maintenance\.venv\Scripts\python.exe` (prefix inside repo — project-local environment rule satisfied).
- Test run (this environment, artifacts present): **49 passed, 0 failed, 0 skipped, 3 warnings**
  - `joblib/numpy_pickle.py` DeprecationWarning (NumPy 2.5 shape deprecation)
  - `xgboost/sklearn.py` UserWarning (joblib-format model loaded via UBJSON fallback)
- FD001 raw data present (`data/raw/train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`). FD002–FD004 absent.

## 2. Known issues confirmed by inspection

### Issue 1 — RMSE 2.402 headline is a capped-target result (CONFIRMED)
`experiments/FD001_final_test_results.json` contains both:
- `metrics_clipped_at_45`: RMSE 2.402, MAE 1.455, R² 0.962, NASA 14.77 (vs clipped truth `min(rul,45)`)
- `metrics_raw_rul`: RMSE **50.375**, MAE **39.055**, R² **-0.469**, NASA **19261.8** (vs raw official RUL)

README leads with the capped number (`RMSE 2.40 | MAE 1.46 | R² 0.962 | NASA score 14.8`), presented as the project headline without the target-truncation caveat up front.

### Issue 2 — RUL cap was optimized by changing the target (CONFIRMED)
Phase 8 (B ablation) swept cap: none→150→125→100→75→60→45 and selected cap 45 by lowest RMSE.
`reports/phase8_ablation.md` itself acknowledges the mechanical component ("a smaller cap shrinks target variance ... part of the gain is mechanical") but still selects cap 45. Comparing RMSE across different target ranges is not a valid model comparison; must not be repeated for the primary V2 benchmark.

### Issue 3 — V2 primary target must be raw RUL (PLANNED)
No raw-RUL benchmark exists yet; all primary tables (`README.md`, `reports/phase8_ablation.md`, `reports/tables/ablation_results.csv`) use capped targets.

### Issue 4 — Validation uses thousands of overlapping windows (CONFIRMED)
`scripts/run_experiment.py` scores every sliding window on validation engines (thousands of samples per engine). The official C-MAPSS test task is one terminal prediction per engine. V2 must replace this with a fixed pseudo-test manifest (one set of terminal cutoffs, identical for every model).

### Issue 5 — NASA scores compared across different sample counts (CONFIRMED)
`reports/phase8_ablation.md` NASA totals are sums over window-dependent sample counts (e.g. w30 vs w120 produce very different numbers of validation windows), so totals are not comparable across rows. V2 must evaluate every configuration on the same fixed manifest.

### Issue 6 — Short test engines padded with zeros, unpadded in training (CONFIRMED)
`scripts/final_evaluation.py:_test_windows` left-pads units shorter than the window (26 of 100 FD001 test units with window 90). Training used only full-length windows, so the padded representation never appeared in training. V2 must make training and inference history handling consistent (e.g. include padded examples + masking, or variable-history features).

### Issue 7 — "test contacted exactly once" is no longer true (CONFIRMED)
The official test labels were evaluated in Phase 9 and the predictions were reused by serving/batch verification (Phase 10, `tests/test_inference_golden.py`). Claims to correct:
- `README.md:5,16,17,37,78` ("exactly once", "one-time")
- `CHANGELOG.md:23,26` ("ONE-TIME", "evaluated exactly once")
- `reports/phase9_final_evaluation.md:9` ("contacted exactly once")
- `configs/final_model.yaml:31` ("CONTACTED EXACTLY ONCE")
- `scripts/final_evaluation.py:6,91,144,179` (ONE-TIME wording)
- `reports/phase10_serving.md:14` (one-time reference)

FD001 official results must be labeled **post-hoc official benchmark** in V2.

### Issue 8 — Original repository attribution missing (CONFIRMED)
No mention anywhere of `aun151214/predictive-maintenance-cmapss`. No Acknowledgements/References section. Grep for `aun151214|predictive-maintenance-cmapss|Acknowledg` returned nothing.

### Issue 9 — Skipped project phases (CONFIRMED)
PROJECT_SPEC roadmap phases 10–16 (error analysis, explainability, uncertainty, Streamlit, CI, FD004, final docs) were replaced by serving work (`reports/phase10_serving.md`, `scripts/serve.py`). These remain to be built under V2. Note: `requirements.txt` already lists `shap` and `streamlit` although neither is implemented, and neither is installed in `.venv` (confirmed via pip list).

### Issue 10 — Clean-checkout test behavior (CONFIRMED)
Tests skip on missing artifacts:
- `tests/test_loader.py` (5 tests) skip without `data/raw`
- `tests/test_artifacts.py` (3 tests) skip without `data/processed`
- `tests/test_inference_golden.py` (2 tests) skip without processed artifacts + model

→ 10 skips on a clean checkout (49 → 39 passed). Documentation does not currently distinguish artifact-free unit tests from dataset/model integration tests, and README claims "49 tests" unconditionally.

### Issue 11 — Documentation cleanup items (CONFIRMED, plus new)
- README `# 0. Environment (Python 3.12 required; TF needs <= 3.12)` vs PROJECT_SPEC `Target Python: 3.11` / `Used: 3.12` vs `pyproject.toml requires-python = ">=3.11"` — inconsistent policy.
- SHAP/Streamlit listed in `requirements.txt` before implementation (Issue 9).
- Notebook `01_data_exploration.ipynb` cell output exposes `D:\Desktop\test\cmapss-rul-predictive-maintenance` (anchored project root).
- `reports/phase8_ablation.md:18` typo "6918 engines" — actually sequences/windows (all windows, not engines; FD001 has 100 training engines).
- Misleading "RMSE 2.40" headline (Issue 1) and "exactly once" language (Issue 7).
- **NEW**: `reports/phase9_final_evaluation.md` contains mojibake (`—?"` for em-dashes) — encoding corruption to fix in the documentation phase.
- **NEW**: `requirements-lock.txt` includes editable install line with a local absolute path (`-e d:\desktop\test\...`) — must be regenerated to a relative/path-free form.
- `experiments/FD001_final_test_results.json` artifacts field stores absolute local paths (machine-specific; acceptable as historical record but not for reproduction).

## 3. Legacy artifact inventory (to preserve, not delete)

- Config: `configs/final_model.yaml` (cap 45, window 90) — copied to `configs/legacy_cap45_model.yaml` for labeling; original retained until references are updated.
- Reports: `reports/phase8_ablation.md`, `reports/phase9_final_evaluation.md`, `reports/phase10_serving.md`, `reports/tables/ablation_results.csv` — labeled as legacy maintenance-horizon (cap-45) experiment in `reports/legacy/README.md`; files not moved to avoid breaking references.
- Results: `experiments/results.csv`, `experiments/FD001_final_test_results.json`, `data/processed/FD001_w90_c45_all/FD001_test_predictions.csv`, `models/final/FD001_final_model.joblib`, `models/checkpoints/*`.
- Split: `experiments/splits/FD001_seed42.json` (80/20) — legacy; V2 will create a new 70/15/15 manifest without overwriting this.

## 4. V2 plan (for reference; NOT executed in this phase)

1. New engine-level split: 70 train / 15 validation / 15 calibration, seed 42 (`experiments/splits/FD001_v2_seed42.json`).
2. Fixed pseudo-test manifests at lifecycle fractions 0.50/0.65/0.80/0.90/0.95 → 75 validation + 75 calibration terminal prediction points.
3. Primary target = RAW RUL; cap 45 demoted to a labeled secondary maintenance-horizon experiment.
4. Consistent training/inference history handling (padding + masking or variable-history features).
5. Raw-RUL benchmark (mean, linear, RF, XGBoost, LSTM, GRU, TCN), window ablations, freeze `configs/final_model_v2_2_fd001.yaml`, post-hoc FD001 evaluation.
6. Error analysis, SHAP explainability, conformal uncertainty (calibration engines only).
7. Streamlit dashboard; keep HTTP serving.
8. Test markers (unit/integration/artifact), GitHub Actions CI, dependency cleanup, consistent Python-version policy.
9. FD004 generalization (raw data absent today; verify sealed status before any read).
10. Documentation rewrite + attribution (`aun151214/predictive-maintenance-cmapss`, NASA C-MAPSS).

## 5. Phase V2-0 changes made

- `AUDIT_V2.md` (this file)
- `configs/legacy_cap45_model.yaml` (preserved frozen config, labeled legacy)
- `reports/legacy/README.md` (labels the cap-45 experiment and its artifacts as the legacy maintenance-horizon task)

No code, no methodology, no results were modified.

## 6. Resolutions (Phase V2-10, CI/deps)

- Issue 9 (SHAP/Streamlit declared but missing): both now installed (`shap==0.52.0`, `streamlit==1.61.1`) and implemented (V2-7, V2-9).
- Issue 11 (Python-version policy): `README.md` now reads "Python >= 3.11, tested on 3.12", consistent with `pyproject.toml requires-python = ">=3.11"` and the documented PROJECT_SPEC deviation (3.12 used).
- Issue 11 (`requirements-lock.txt` absolute editable path): lock regenerated (2026-08-15 snapshot, includes shap/streamlit); the editable line is replaced by a path-free comment — install the package via `pip install -e . --no-deps`.
- Issue 10 (clean-checkout test behavior): `tests/test_v2_serving.py` is now marked `needs_artifacts`; the marker is registered in `pyproject.toml`; CI (`.github/workflows/ci.yml`) runs `pytest -m "not needs_artifacts"`. Verified on an artifact-free tree: 78 passed, 10 skipped, 3 deselected.