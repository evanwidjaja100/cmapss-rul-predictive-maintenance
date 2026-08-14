# CHANGELOG

## [0.6.0] — Phase 5 — 2026-08-14

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