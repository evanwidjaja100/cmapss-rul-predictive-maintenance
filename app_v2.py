"""Streamlit serving app for the frozen V2.2 model (Methodology V2.2).

Run:  .venv\\Scripts\\python.exe -m streamlit run app_v2.py

Serves the deployment model selected by the pre-registered V2.2 policy
(configs/final_model_v2_2_fd001.yaml) with the recalibrated engine-cluster
conformal interval. Shows model version, predicted raw RUL, observed cycles,
history_is_padded + padded timestep count, the conformal interval and the
calibration method. No OOD classification: `history_is_padded` is objective,
`short_history_risk_flag` is an empirical flag (reports/v2_2_error_analysis.md).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from rul_prediction.data.loader import DATA_COLUMNS, load_test
from rul_prediction.serving.v2_predictor import (
    RISK_OBSERVED_CYCLES,
    V2Predictor,
    limited_history_warning,
)

RISK_SENSORS = ["sensor_4", "sensor_9", "sensor_11", "sensor_12", "sensor_3"]


@st.cache_resource
def get_predictor() -> V2Predictor:
    return V2Predictor()


predictor = get_predictor()

st.set_page_config(page_title=f"CMAPSS RUL — {predictor.model_version}", page_icon="🔧", layout="wide")
st.title("CMAPSS FD001 remaining-useful-life predictor")
st.caption(
    f"Methodology V2.2 deployment model: {predictor.model_version}, window "
    f"{predictor.window}, raw RUL target. 90% engine-cluster conformal interval "
    f"(q = {predictor.q_cycles:.1f} cycles, 15 calibration engines). "
    "Post-hoc: official test labels were inspected in the V2-0 audit."
)
st.info(predictor.uncertainty_disclosure)

with st.sidebar:
    st.header("Method notes (V2.2)")
    st.write(
        f"- Target: **raw RUL** (cycles to failure, uncapped).\n"
        f"- 90% interval: engine-cluster finite-sample conformal guarantee "
        f"(k = ceil((n+1)(1-alpha)) with n = 15 calibration engines; one "
        "maximum-error score per engine across five predefined lifecycle "
        "checkpoints). Formal guarantee only under exchangeability of engines "
        "with the predefined checkpoint scheme.\n"
        f"- {predictor.calibration_method}\n"
        f"- Test trajectories are truncated before failure: `cycle.max()` is the "
        "observed history length, NOT a lifetime.\n"
        f"- `history_is_padded` = observed cycles < window ({predictor.window}); "
        "the window is left-padded in the shared representation - expected "
        "input, not OOD.\n"
        f"- `short_history_risk_flag` (observed < {RISK_OBSERVED_CYCLES}) is an "
        "EMPIRICAL risk flag from the V2.2 error analysis "
        "(reports/v2_2_error_analysis.md), not an OOD classification.\n"
        f"- Model: `models/v2_2/fd001_{predictor.candidate}.{'joblib' if predictor._model_name in ('rf','xgboost') else 'keras'}`; "
        "CV + selection + freeze in reports/v2_2_methodology.md."
    )

mode = st.radio("Input", ["Demo engine (official FD001 test)", "Upload C-MAPSS file"])

if mode.startswith("Demo"):
    test = load_test("FD001")
    engine = st.selectbox("Engine", sorted(test["engine_id"].unique()),
                          format_func=lambda e: f"engine {e}")
    history = test[test["engine_id"] == engine].sort_values("cycle")
    result = predictor.predict_frame(history).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted RUL (cycles)", f"{result['prediction_raw_rul']:.0f}")
    c2.metric("90% interval", f"[{result['lo_90']:.0f}, {result['hi_90']:.0f}]")
    c3.metric("Observed cycles", f"{result['n_cycles_observed']}")
    c4.metric("Model version", result["model_version"])
    warning = limited_history_warning(int(result["n_cycles_observed"]), predictor.window)
    if warning:
        st.warning(warning)
    if result["history_is_padded"]:
        st.info(f"Padded window: {int(result['n_padded_timesteps'])} of "
                f"{predictor.window} timesteps padded (expected input, not OOD).")
    if result["short_history_risk_flag"]:
        st.info(f"Empirical risk flag: {int(result['n_cycles_observed'])} observed cycles "
                f"< {RISK_OBSERVED_CYCLES}; overprediction concentrates in this group "
                "(reports/v2_2_error_analysis.md).")
    st.subheader("Top V2.2 sensitivity sensors")
    st.line_chart(history[["cycle", *RISK_SENSORS]].set_index("cycle"))
else:
    uploaded = st.file_uploader("C-MAPSS test-style file (columns: engine_id, cycle, "
                                "setting_1..3, sensor_1..21)", type=["txt", "csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded, sep=r"\s+", header=None, engine="python")
        frame.columns = DATA_COLUMNS[: frame.shape[1]]
        table = predictor.predict_frame(frame)
        n_padded = int(table["history_is_padded"].sum())
        n_risk = int(table["short_history_risk_flag"].sum())
        st.dataframe(table)
        if n_padded:
            st.warning(f"{n_padded} of {len(table)} engines have padded windows "
                       f"(observed < {predictor.window} cycles); windows are left-padded.")
        if n_risk:
            st.info(f"{n_risk} of {len(table)} engines carry the empirical short-history "
                    f"risk flag (observed < {RISK_OBSERVED_CYCLES} cycles).")
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv, "v2_2_predictions.csv", "text/csv")