# Third-Party Notices

## Upstream project

- **Project:** `aun151214/predictive-maintenance-cmapss`
  (https://github.com/aun151214/predictive-maintenance-cmapss)
- **Role:** this repository is a maintained continuation of the methodology
  and repository structure of the upstream project (see README
  "Origin project"). The upstream Phase 1–10 cap-45 experiment is preserved
  and explicitly labeled as a legacy maintenance-horizon task.
- **Source code copied:** none. All code in this repository is written by
  this project; the upstream repository was used as a structural/methodological
  reference, and its license terms (MIT) are acknowledged above.
- **Data:** NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set
  (public dataset distributed by NASA; files included under `data/raw/` are
  gitignored and not redistributed here).

## Direct dependencies

Runtime dependencies are declared in `pyproject.toml` /
`requirements-lock.txt` (numpy, pandas, scipy, scikit-learn, matplotlib,
pyyaml, joblib, tensorflow, keras, xgboost, shap, streamlit). Each is
distributed under its own open-source license; see the respective package
metadata. No vendored copies of these dependencies are included in this
repository.