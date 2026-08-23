> **SUPERSEDED BY METHODOLOGY M2** — this report documents the M1
> methodology as historical record. M2 corrects the lifetime semantics
> (observed history vs implied failure cycle), replaces the single 15-engine
> validation with 5-fold engine-group CV, recalibrates conformal at the
> engine level, and fixes the FD004 collapse with per-regime scaling.
> See M2_REPAIR_PLAN.md and reports/m2_methodology.md.

# Phase M1-9 — Streamlit Serving App

Date: 2026-08-15  |  Served model: frozen GRU w45 huber (`models/m1_frozen_gru_w45_huber.keras`)  |  Target: raw RUL

## App

`app_m1.py` — run with `.venv\Scripts\python.exe -m streamlit run app_m1.py`.

- **Demo mode:** pick any official FD001 test engine; shows the raw-RUL prediction, the 90% conformal interval `[ŷ−q, ŷ+q]` (q = 24.1 cycles, M1-8), the alarm lower bound, observed-cycle count, and an OOD warning for short histories; plots the five risk-relevant sensors (2, 4, 6, 7, 8 — M1-7 finding) over the engine trajectory.
- **Upload mode:** any C-MAPSS-style text file (engine_id, cycle, settings, sensors) → per-engine prediction table with intervals and OOD flags, downloadable as CSV; a warning summarizes how many engines are OOD (measured coverage there is 47.7%, M1-8).
- **Honesty in the UI:** caption states the post-hoc status (official labels inspected in the M1-0 audit) and the method sidebar documents the 88% validation coverage and the OOD caveat.

## Serving core

`src/rul_prediction/serving/m1_predictor.py` — `M1Predictor.predict_frame(frame)`:

- reuses the exact freeze inference path (`make_predictor` + train-only scaler + shared window builder) — the serving predictions are **bit-identical to the Phase M1-5 freeze** (tests pin engine 67 → 187.640411 and engine 78 → 196.238007, matching `experiments/m1_fd001_official_predictions.csv`);
- reads q for α=0.1 from `reports/tables/m1_conformal_calibration.csv` (single source of truth);
- flags `ood_short_history` for engines with < 128 observed cycles (training lifetime minimum, M1-6).

## Tests

`tests/test_m1_serving.py` (3 tests): freeze-prediction reproduction, interval/OOD math (q ≈ 24.10, 44 OOD engines on the official test, 4 padded engines), all passing. Suite total: **91 tests**.

## Notes

- `streamlit` 1.61.1 now installed — the second M1-0 audit finding (declared-but-missing dependency) is resolved; `requirements.txt` already listed both `shap` and `streamlit`.
- The legacy phase-10 serving (`scripts/serve.py`, xgboost cap-45 pipeline) is untouched and remains the legacy pipeline; the M1 app is the raw-RUL GRU service. Both coexist; the M1 model is the frozen deliverable of this methodology.

Next: **Phase M1-10 — CI / test hardening / dependency audit** (reconcile requirements-lock absolute path, Python-version policy, test determinism).