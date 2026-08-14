# CHANGELOG

## [0.4.0] — Phase 3 — 2026-08-14

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