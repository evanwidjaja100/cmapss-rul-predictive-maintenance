# Phase 10 — Deployment-Ready Inference Service

Date: 2026-08-15

## Deliverables

- `src/rul_prediction/serving/inference.py` — `RulPredictor`: loads the frozen `configs/final_model.yaml` (validated against the frozen values: model/variant/window/max_rul must match exactly), the Phase 9 model and the train-only scaler, and predicts per-unit RUL with the identical Phase 9 pipeline (scale → 90-cycle windows, zero-pad short units → 169 engineered features → predict → clip [0, 45]).
- `scripts/serve.py` — two modes:
  - `serve --port 8000`: stdlib `http.server` (no new dependencies). `GET /health` returns model/variant/max_rul; `POST /predict` accepts JSON `{"rows": [[engine_id, cycle, setting_1..3, sensor_1..21], ...]}` (C-MAPSS column order, positional columns accepted) and returns per-unit predictions with padding flags.
  - `batch --input <test-style .txt> [--rul RUL_FD001.txt]`: writes `experiments/serving_predictions.csv`; prints RMSE/MAE/R²/NASA when ground truth is given.

## Acceptance checks

- **Golden-file test** (`tests/test_inference_golden.py`): serving predictions over the full official test file match the Phase 9 evaluation (`FD001_test_predictions.csv`) per unit within 1e-4. This is the anti-regression guarantee that serving never drifts from the frozen evaluation. (The Phase 9 evaluation is a **post-hoc official benchmark** in V2 terminology — labels were re-inspected during V2 audits; see `AUDIT_V2.md` Issue 7.)
- Short-unit padding re-verified (shortest test unit → padded flag).
- HTTP smoke test: `GET /health` OK; `POST /predict` with the full 13,096-row test file → 100 units, 26 padded, predictions consistent with Phase 9.
- Batch mode on the official test file reproduces Phase 9 metrics exactly: **RMSE 2.4024 | MAE 1.4548 | R² 0.9619 | NASA 14.767**.
- 49 pytest tests pass.

## Notes

- No retraining, no test-set contact beyond the Phase 9 evaluation (the golden test compares against the already-committed Phase 9 predictions; the model file is loaded, never refit).
- The model artifact is `models/final/FD001_final_model.joblib` (gitignored, regenerable via `scripts/final_evaluation.py`); xgboost logs a benign "unknown file format" warning when loading it (UBJSON content under a `.joblib` name) — cosmetic only.
- Deployment scope: single local process, thread-safe predictor (stateless; `ThreadingHTTPServer`). Skipped: auth, TLS, Docker, Kubernetes, model registry — add when a real deployment target exists (YAGNI).