import datetime as dt
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import config as C
from features import prepare


@st.cache_resource(show_spinner="Loading trained pipeline...")
def load_bundle():
    """Loads the saved preprocessing + model pipeline once (never retrains on refresh)."""
    return joblib.load(C.MODEL_PATH)


def predict_df(df: pd.DataFrame) -> pd.DataFrame:
    b = load_bundle()
    X = df[C.FEATURES]
    out = df.copy()
    out["Breach_Probability"] = b["pipeline"].predict_proba(X)[:, 1]
    out["Predicted_Breach"] = b["pipeline"].predict(X)
    return out


def render():
    b = load_bundle()
    meta = b["metadata"]
    st.header("Prediction")
    st.caption(f"Model: {meta['model_name']} | Version: {meta['version']} | Trained: {meta['trained_at']}")

    tab1, tab2 = st.tabs(["Single request", "Model metrics"])

    with tab1:
        cats = meta["categories"]
        with st.form("predict_form"):
            c1, c2, c3 = st.columns(3)
            st_ = c1.selectbox("Service type", cats["Service_Type"])
            rq = c2.selectbox("Requester type", cats["Requester_Type"])
            ch = c3.selectbox("Requested channel", cats["Requested_Channel"])
            pr = c1.selectbox("Priority", cats["Priority"])
            office = c2.selectbox("Assigned office", cats["Assigned_Office"])
            docs = c3.selectbox("Required documents complete?", cats["Required_Documents_Complete"],
                                index=cats["Required_Documents_Complete"].index("Yes"))
            ql = c1.number_input("Queue length at submission", 0, 200, 12)
            sla = c2.number_input("SLA days", 1, 30, 3)
            sd = c3.date_input("Submission date", dt.date.today())
            go = st.form_submit_button("Predict")
        if go:
            raw = pd.DataFrame([{"Service_Type": st_, "Requester_Type": rq, "Submission_Date": str(sd),
                                 "Requested_Channel": ch, "Priority": pr, "Assigned_Office": office,
                                 "Queue_Length": ql, "Required_Documents_Complete": docs, "SLA_Days": sla}])
            row = prepare(raw)[C.FEATURES]
            p = float(b["pipeline"].predict_proba(row)[0, 1])
            label = "SLA BREACH LIKELY" if p >= 0.5 else "Likely within SLA"
            (st.error if p >= 0.5 else st.success)(f"**{label}**  |  Breach probability: **{p:.1%}**")
            now = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S}"
            st.caption(f"Prediction timestamp: {now} | Model version {meta['version']}")
            st.warning("Decision support only. Student services staff must review this result before any "
                       "action; it never authorizes denying or delaying a service.")
            st.session_state.setdefault("pred_log", []).append(
                {**raw.iloc[0].to_dict(), "Breach_Probability": round(p, 3), "Label": label, "Timestamp": now})
        if st.session_state.get("pred_log"):
            st.subheader("Predictions this session")
            st.dataframe(pd.DataFrame(st.session_state["pred_log"]), width="stretch")

    with tab2:
        m = meta["metrics"]
        a, b2, c, d = st.columns(4)
        a.metric("Accuracy", f"{m['accuracy']:.3f}"); b2.metric("Precision", f"{m['precision']:.3f}")
        c.metric("Recall", f"{m['recall']:.3f}"); d.metric("F1", f"{m['f1']:.3f}")
        cm = pd.DataFrame(m["confusion_matrix"], index=["Actual: Within", "Actual: Breach"],
                          columns=["Pred: Within", "Pred: Breach"])
        st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                  title="Confusion matrix (test set)"), width="stretch")
        fi = pd.Series(meta["feature_importance"]).sort_values().tail(12).reset_index()
        fi.columns = ["Feature", "Importance"]
        st.plotly_chart(px.bar(fi, x="Importance", y="Feature", orientation="h",
                               title="Top feature importances"), width="stretch")
        st.caption(f"Trained on {meta['n_train']} / tested on {meta['n_test']} completed requests "
                   f"(breach rate {meta['breach_rate']:.0%}). Target: {meta['target']}. "
                   "5-fold CV F1 on training data: " + ", ".join(f"{k} {v:.3f}" for k, v in meta["cv_f1"].items()))
        st.markdown("**Limitations**")
        st.markdown("- Small dataset (about 350 labeled requests); metrics have wide uncertainty.\n"
                    "- Probabilities come from a Random Forest and are not calibrated guarantees.\n"
                    "- The model predicts risk of delay, not the reason; staff judgment is required.\n"
                    "- Retrain periodically; workload and policy changes cause drift.")
