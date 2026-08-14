# CHANGELOG

## [0.2.0] — Phase 1 — 2026-08-14

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