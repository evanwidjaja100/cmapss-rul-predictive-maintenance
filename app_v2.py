"""Streamlit serving app for the frozen V2 model (Methodology V2, Phase V2-9).

Run:  .venv\\Scripts\\python.exe -m streamlit run app_v2.py

Serves the frozen GRU w45 huber model (raw RUL) with 90% conformal intervals
(V2-8) and the short-history out-of-distribution flag (V2-6). Demo mode uses
the official FD001 test engines; upload mode accepts C-MAPSS-style files.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from rul_prediction.data.loader import DATA_COLUMNS, load_test
from rul_prediction.serving.v2_predictor import OOD_LIFETIME, V2Predictor

RISK_SENSORS = ["sensor_2", "sensor_4", "sensor_6", "sensor_7", "sensor_8"]


@st.cache_resource
def get_predictor() -> V2Predictor:
    return V2Predictor()


st.set_page_config(page_title="CMAPSS RUL — frozen V2 model", page_icon="🔧", layout="wide")
st.title("CMAPSS FD001 remaining-useful-life predictor")
st.caption(
    f"Frozen V2 model: GRU, window 45, huber loss, raw RUL target (no cap). "
    f"90% split-conformal interval (q = {get_predictor().q_cycles:.1f} cycles) and OOD flag per engine. "
    "Post-hoc: official test labels were inspected in the V2-0 audit."
)

with st.sidebar:
    st.header("Method notes")
    st.write(
        f"- Target: **raw RUL** (cycles to failure, uncapped).\n"
        f"- 90% interval: finite-sample conformal guarantee on exchangeable data "
        "(validation coverage measured 88%).\n"
        f"- Engines with fewer than {OOD_LIFETIME} observed cycles are flagged **OOD**: "
        "below the training lifetime minimum; measured coverage there is 47.7% "
        "(reports/v2_conformal.md).\n"
        "- Model: `models/v2_frozen_gru_w45_huber.keras`; freeze + post-hoc "
        "evaluation in reports/v2_freeze.md."
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
    c4.metric("Observed cycles", f"{result['n_cycles']}")
    if result["ood_short_history"]:
        st.warning(f"OOD: observed history ({result['n_cycles']} cycles) is below the "
                   f"training minimum ({OOD_LIFETIME}); conformal coverage is unreliable here.")
    st.subheader("Risk-relevant sensors (V2-7)")
    st.line_chart(history[["cycle", *RISK_SENSORS]].set_index("cycle"))
else:
    uploaded = st.file_uploader("C-MAPSS test-style file (columns: engine_id, cycle, "
                                "setting_1..3, sensor_1..21)", type=["txt", "csv"])
    if uploaded is not None:
        frame = pd.read_csv(uploaded, sep=r"\s+", header=None, engine="python")
        frame.columns = DATA_COLUMNS[: frame.shape[1]]
        table = predictor.predict_frame(frame)
        n_ood = int(table["ood_short_history"].sum())
        st.dataframe(table)
        if n_ood:
            st.warning(f"{n_ood} of {len(table)} engines are OOD (short history < {OOD_LIFETIME} "
                       f"cycles): treat their intervals with caution (measured coverage 47.7%).")
        csv = table.to_csv(index=False).encode("utf-8")
        st.download_button("Download predictions CSV", csv, "v2_predictions.csv", "text/csv")