# PROJECT_SPEC.md

Leakage-Safe Remaining Useful Life Prediction for Predictive Maintenance Using NASA C-MAPSS

## 1. Problem

Predict the Remaining Useful Life (RUL) of turbofan engines from multivariate sensor time series using the NASA C-MAPSS dataset. RUL is defined as the number of operational cycles remaining before engine failure.

## 2. Dataset

- Initial dataset: **FD001** (single operating condition, one fault mode, 100 training / 100 test engines)
- Future datasets: FD002, FD003, FD004
- Raw files (FD001): `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`

## 3. Validation methodology

Training dataset engines are divided by engine ID into:

- **Training engines** (≈80 %): model fitting and preprocessing fitting
- **Validation engines** (≈20 %): model selection, early stopping, hyperparameter tuning

The **official test set is used for final evaluation only**. It is never used for model selection, early stopping, hyperparameter tuning, normalization fitting, feature selection, architecture selection, RUL-clipping selection, sequence-length selection, loss-function selection, or threshold optimization.

Splitting is performed at engine level; overlapping sequence windows from one engine never cross the train/validation boundary.

## 4. Metrics

- RMSE
- MAE
- R²
- NASA / PHM asymmetric score

All metrics are computed programmatically from predictions; no metric is fabricated.

## 5. Non-negotiable rules

1. Test data cannot influence model configuration.
2. Test data cannot be passed to `model.fit()` as `validation_data`.
3. Scalers are fitted using training engines only.
4. Sequence windows from the same engine cannot appear in both training and validation.
5. Hyperparameters are selected using validation engines only.
6. Official test results are generated only after model configuration is frozen.
7. Experiments use reproducible random seeds.
8. Metrics are always generated programmatically.
9. Results must never be fabricated.
10. Experimental methodology must not be changed silently; any change becomes a new documented experimental cycle.

## 6. Environment

- Target Python: 3.11
- **Used Python: 3.12** — deviation documented: Python 3.11 is not installed on this machine, and the earliest TensorFlow release supporting Python 3.14 does not exist, so Python 3.12 (officially supported by TensorFlow starting with TF 2.16) is used. No other behavior changes.
- All libraries are installed inside `<project>/.venv`. Global pip is not used.

## 7. Roadmap

| Phase | Scope |
|---|---|
| 0 | Project initialization and methodology |
| 1 | Data acquisition and validation |
| 2 | Exploratory data analysis |
| 3 | Leakage-safe data splitting |
| 4 | RUL preprocessing and sequence generation |
| 5 | Classical ML baselines |
| 6 | LSTM and GRU baselines |
| 7 | TCN improvement model |
| 8 | Ablation and hyperparameter experiments |
| 9 | Freeze configuration and final test |
| 10 | Engineering error analysis |
| 11 | Explainable AI |
| 12 | Prediction uncertainty |
| 13 | Streamlit dashboard |
| 14 | Testing, reproducibility and CI |
| 15 | FD004 generalization study |
| 16 | Final README and portfolio presentation |

## 8. Reproducibility

Every experiment records: random seed, dataset, train engine IDs, validation engine IDs, sequence length, RUL cap, features, scaler, model, model configuration, optimizer, learning rate, loss, batch size, epochs, early-stopping settings, training time, validation metrics. Default deterministic seed: **42**.

## 9. Governance

- Phases are completed one at a time; each phase ends with tests, acceptance-criteria verification, a commit, and a STOP.
- A new phase begins only on explicit instruction.
- Failed experiments are reported, never hidden. Test performance is reported honestly even when disappointing.