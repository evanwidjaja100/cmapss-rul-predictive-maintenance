"""Streamlit serving app for the frozen V2.1 model (Methodology V2.1).

Run:  .venv\\Scripts\\python.exe -m streamlit run app_v2.py

Serves the CV-selected GRU w45 huber model (raw RUL) with the engine-cluster
conformal interval (engine-level, 15 calibration engines) and objective /
empirical history flags - no OOD classification (V2_1_REPAIR_PLAN.md R3/R12).
Demo mode uses the official FD001 test engines; upload mode accepts
C-MAPSS-style files.
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

RISK_SENSORS = ["sensor_2", "sensor_4", "sensor_6", "sensor_7", "sensor_8"]


@st.cache_resource
def get_predictor() -> V2Predictor:
    return V2Predictor()


st.set_page_config(page_title="CMAPSS RUL — frozen V2.1 model", page_icon="🔧", layout="wide")
st.title("CMAPSS FD001 remaining-useful-life predictor")
st.caption(
    f"Frozen V2.1 model: GRU, window 45, huber loss, raw RUL target (no cap). "
    f"90% engine-cluster conformal interval (q = {get_predictor().q_cycles:.1f} cycles, "
    "15 calibration engines). Post-hoc: official test labels were inspected in the V2-0 audit."
)

with st.sidebar:
    st.header("Method notes")
    st.write(
        f"- Target: **raw RUL** (cycles to failure, uncapped).\n"
        f"- 90% interval: engine-cluster finite-sample conformal guarantee "
        f"(k = ceil((n+1)(1-alpha)) with n = 15 calibration engines); empirical "
        "official-test coverage 98% at alpha=0.1 (reports/v2_1_conformal.md).\n"
        f"- Test trajectories are truncated before failure: `cycle.max()` is the "
        "observed history length, NOT a lifetime.\n"
        f"- `history_is_padded` = observed cycles < window (45); the window is "
        "left-padded in the shared representation - expected input, not OOD.\n"
        f"- `short_history_risk_flag` (observed < {RISK_OBSERVED_CYCLES}) is an "
        "EMPIRICAL risk flag from the V2.1 error analysis, not an OOD "
        "classification (reports/v2_1_error_analysis.md).\n"
        "- Model: `models/v2_1/fd001_gru_w45_huber.keras`; CV + freeze + post-hoc "
        "evaluation in reports/v2_1_cross_validation.md."
    )

predictor = get_predictor()
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
    c3.metric("Alarm lower bound", f"{result['alarm_lower_bound']:.0f}")
    c4.metric("Observed cycles", f"{result['n_cycles_observed']}")
    warning = limited_history_warning(int(result["n_cycles_observed"]))
    if warning:
        st.warning(warning)
    if result["short_history_risk_flag"]:
        st.info(f"Empirical risk flag: {int(result['n_cycles_observed'])} observed cycles "
                f"< {RISK_OBSERVED_CYCLES}; overprediction concentrates in this group "
                "(see reports/v2_1_error_analysis.md).")
    st.subheader("Risk-relevant sensors (V2-7)")
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
                       f"(observed < 45 cycles); their windows are left-padded.")
        if n_risk:
            st.info(f"{n_risk} of {len(table)} engines carry the empirical short-history "
                    f"risk flag (observed < {RISK_OBSERVED_CYCLES} cycles).")
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv, "v2_1_predictions.csv", "text/csv")