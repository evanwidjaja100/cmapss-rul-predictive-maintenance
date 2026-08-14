# cmapss-rul-predictive-maintenance

**Leakage-Safe Remaining Useful Life Prediction for Predictive Maintenance Using NASA C-MAPSS**

A professional, reproducible RUL prediction system for turbofan-engine prognostics, built phase-by-phase with strict leakage safety: engine-level validation splits, training-engine-only preprocessing fitting, and an official test set that is never touched during model development.

## Status

Phase 0 — Project initialization and methodology (in progress).

## Scientific guardrails

1. Official test data never influences model configuration.
2. Test data never used as Keras `validation_data`.
3. Scalers fitted on training engines only.
4. Windows from one engine never split across train/validation.
5. Hyperparameters chosen on validation engines only.
6. Test evaluation happens only after configuration is frozen.
7. Reproducible seeds; metrics always computed programmatically.
8. No fabricated results; methodology changes are documented.

See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full specification.

## Environment (Windows)

```bash
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All commands use `.venv\Scripts\python.exe`; never global pip.

## Tests

```bash
.venv\Scripts\python.exe -m pytest
```

## License

MIT — see [LICENSE](LICENSE).