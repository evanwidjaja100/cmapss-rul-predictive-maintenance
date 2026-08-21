"""Streamlit serving app for the frozen V2.2 model (Methodology V2.2).

Run:  .venv\\Scripts\\python.exe -m streamlit run app_v2.py

Serves the deployment model selected by the pre-specified V2.2 policy
(configs/final_model_v2_2_fd001.yaml) with the engine-cluster conformal
interval (q from the tracked deployment config
configs/deployment_v2_2_fd001.yaml). Shows model version, predicted raw RUL,
observed cycles, the objective padding/history fields (history_is_padded +
n_padded_timesteps), the conformal interval and the calibration method. No OOD
classification and no empirical risk threshold: `history_is_padded` is an
objective input-representation fact.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from rul_prediction.data.loader import DATA_COLUMNS, load_test

RISK_SENSORS = ["sensor_4", "sensor_9", "sensor_11", "sensor_12", "sensor_3"]


@st.cache_resource
def get_predictor():
    # ponytail: lazy import inside function, not at module import; keeps `import app_v2` artifact-free
    from rul_prediction.serving.v2_predictor import V2Predictor

    return V2Predictor()


def main() -> None:
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
            f"- 90% interval: engine-cluster finite-sample conformal interval "
            f"(k = ceil((n+1)(1-alpha)) with n = 15 calibration engines; one "
            "maximum-error score per engine across five predefined lifecycle "
            "checkpoints). Calibration engines were inspected during earlier "
            "project iterations, so the interval is empirically calibrated, not "
            "a pristine one-shot external guarantee.\n"
            f"- {predictor.calibration_method}\n"
            f"- Test trajectories are truncated before failure: `cycle.max()` is the "
            "observed history length, NOT a lifetime.\n"
            f"- `history_is_padded` = observed cycles < window ({predictor.window}); "
            "the window is left-padded in the shared representation - expected "
            "input, not OOD.\n"
            f"- Model: `models/v2_2/fd001_{predictor.candidate}.{'joblib' if predictor._model_name in ('rf','xgboost') else 'keras'}`; "
            "CV + selection + freeze in reports/v2_2_final_report.md."
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
        warning = __import__("rul_prediction.serving.v2_predictor", fromlist=["limited_history_warning"]).limited_history_warning(int(result["n_cycles_observed"]), predictor.window)
        if warning:
            st.warning(warning)
        if result["history_is_padded"]:
            st.info(f"{int(result['n_cycles_observed'])} cycles are observed while the "
                    f"model window is {predictor.window}; "
                    f"{int(result['n_padded_timesteps'])} leading timesteps were padded "
                    "(expected input, not OOD).")
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
            st.dataframe(table)
            if n_padded:
                st.warning(f"{n_padded} of {len(table)} engines have padded windows "
                           f"(observed < {predictor.window} cycles); windows are left-padded.")
            csv = table.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", csv, "v2_2_predictions.csv", "text/csv")


if __name__ == "__main__":
    main()
