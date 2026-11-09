import pandas as pd
import streamlit as st
import config as C
import data_processing as dp
import exploratory_analysis, prediction, dashboard, system_design

st.set_page_config(page_title="SAD Predictive System", page_icon="📊", layout="wide")

menu = st.sidebar.selectbox("Module", ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"])
st.sidebar.caption("Student Service Request SLA Monitoring\nDecision support only; humans decide.")


def active_df() -> pd.DataFrame:
    """Uploaded dataset if valid, otherwise the bundled synthetic sample."""
    if "df" in st.session_state:
        return st.session_state["df"]
    return dp.clean(dp.load_csv(C.SAMPLE_DATA_PATH))


if menu == "Home":
    meta = prediction.load_bundle()["metadata"]
    st.title("Student Service Request SLA Breach Prediction")
    st.write("Deploys the midterm data-mining model as a decision-support web app for student services staff.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model", meta["model_name"].split(" ")[0]); c2.metric("Version", meta["version"])
    c3.metric("Test F1", f"{meta['metrics']['f1']:.3f}")
    st.markdown("""
**Modules:** Dataset (upload & validate) → Data Analysis (EDA) → Prediction (single request + metrics) →
Dashboard (KPIs, review queue, staff decisions) → System Design (requirements, use case, workflow).

**Target:** `SLA_Breach` = 1 when `Processing_Days` > `SLA_Days`.  **Prediction checkpoint:** at submission/triage time,
so `Processing_Days`, `Status`, `Reopened` and `Satisfaction_Score` are excluded to prevent leakage.
""")
    st.warning("Predictions are decision-support outputs, not authorization to deny or delay student services. "
               "Use synthetic or properly authorized data only; do not upload real student records to the public deployment.")

elif menu == "Dataset":
    st.header("Dataset Upload and Validation")
    up = st.file_uploader("Upload CSV", type="csv")
    st.caption("Required columns: " + ", ".join(C.RAW_REQUIRED) + f" (optional: {C.ID_COL}, Status, Processing_Days)")
    if up is not None:
        try:
            raw = dp.load_csv(up)
        except Exception as e:
            st.error(f"Could not read the file as CSV: {e}"); st.stop()
        res = dp.validate_schema(raw)
        for w in res["warnings"]: st.warning(w)
        if not res["ok"]:
            for e in res["errors"]: st.error(e)
            st.stop()
        st.session_state["df"] = dp.clean(raw)
        st.success(f"Dataset accepted and saved for all modules ({len(st.session_state['df'])} clean rows).")
        df = raw
    else:
        st.info("No file uploaded. Showing the bundled synthetic sample.")
        df = dp.load_csv(C.SAMPLE_DATA_PATH)
    q = dp.quality_summary(df)
    a, b, c = st.columns(3); a.metric("Rows", q["rows"]); b.metric("Columns", q["cols"]); c.metric("Duplicate rows", q["duplicates"])
    st.subheader("Preview"); st.dataframe(df.head(20), width="stretch")
    st.subheader("Null summary")
    st.dataframe(q["nulls"], width="stretch") if len(q["nulls"]) else st.success("No missing values.")
    if "df" in st.session_state and st.button("Reset to sample dataset"):
        del st.session_state["df"]; st.rerun()

elif menu == "Data Analysis":
    exploratory_analysis.render(active_df())
elif menu == "Prediction":
    prediction.render()
elif menu == "Dashboard":
    dashboard.render(active_df())
else:
    system_design.render()
