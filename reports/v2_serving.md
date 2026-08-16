# Phase V2-9 — Streamlit Serving App

Date: 2026-08-15  |  Served model: frozen GRU w45 huber (`models/v2_frozen_gru_w45_huber.keras`)  |  Target: raw RUL

## App

`app_v2.py` — run with `.venv\Scripts\python.exe -m streamlit run app_v2.py`.

- **Demo mode:** pick any official FD001 test engine; shows the raw-RUL prediction, the 90% conformal interval `[ŷ−q, ŷ+q]` (q = 24.1 cycles, V2-8), the alarm lower bound, observed-cycle count, and an OOD warning for short histories; plots the five risk-relevant sensors (2, 4, 6, 7, 8 — V2-7 finding) over the engine trajectory.
- **Upload mode:** any C-MAPSS-style text file (engine_id, cycle, settings, sensors) → per-engine prediction table with intervals and OOD flags, downloadable as CSV; a warning summarizes how many engines are OOD (measured coverage there is 47.7%, V2-8).
- **Honesty in the UI:** caption states the post-hoc status (official labels inspected in the V2-0 audit) and the method sidebar documents the 88% validation coverage and the OOD caveat.

## Serving core

`src/rul_prediction/serving/v2_predictor.py` — `V2Predictor.predict_frame(frame)`:

- reuses the exact freeze inference path (`make_predictor` + train-only scaler + shared window builder) — the serving predictions are **bit-identical to the Phase V2-5 freeze** (tests pin engine 67 → 187.640411 and engine 78 → 196.238007, matching `experiments/v2_fd001_official_predictions.csv`);
- reads q for α=0.1 from `reports/tables/v2_conformal_calibration.csv` (single source of truth);
- flags `ood_short_history` for engines with < 128 observed cycles (training lifetime minimum, V2-6).

## Tests

`tests/test_v2_serving.py` (3 tests): freeze-prediction reproduction, interval/OOD math (q ≈ 24.10, 44 OOD engines on the official test, 4 padded engines), all passing. Suite total: **91 tests**.

## Notes

- `streamlit` 1.61.1 now installed — the second V2-0 audit finding (declared-but-missing dependency) is resolved; `requirements.txt` already listed both `shap` and `streamlit`.
- The legacy phase-10 serving (`scripts/serve.py`, xgboost cap-45 pipeline) is untouched and remains the legacy pipeline; the V2 app is the raw-RUL GRU service. Both coexist; the V2 model is the frozen deliverable of this methodology.

Next: **Phase V2-10 — CI / test hardening / dependency audit** (reconcile requirements-lock absolute path, Python-version policy, test determinism).