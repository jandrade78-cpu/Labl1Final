import datetime as dt
import pandas as pd
import plotly.express as px
import streamlit as st
import config as C
from prediction import predict_df, load_bundle


def _band(p, hi, med):
    return "High" if p >= hi else ("Medium" if p >= med else "Low")


def _follow_up(row):
    if row["Risk"] == "Low":
        return "Routine processing"
    tips = []
    if row["Queue_Length"] > 20: tips.append("rebalance queue / reassign staff")
    if row["Required_Documents_Complete"] == "No": tips.append("contact requester for missing documents")
    if row["Priority"] == "High": tips.append("escalate to office head")
    return "; ".join(tips) or "supervisor review before SLA deadline"


def render(df: pd.DataFrame):
    st.header("Decision-Support Dashboard")
    meta = load_bundle()["metadata"]
    hi = st.sidebar.slider("High-risk threshold", 0.5, 0.95, 0.7, 0.05)
    med = st.sidebar.slider("Medium-risk threshold", 0.2, hi, min(0.4, hi), 0.05)

    scope = df
    if "Status" in df.columns:
        if st.sidebar.checkbox("Score only open (Pending) requests", value=True):
            scope = df[df["Status"] == "Pending"]
    if scope.empty:
        st.warning("No requests in scope."); return
    s = predict_df(scope)
    s["Risk"] = s["Breach_Probability"].apply(lambda p: _band(p, hi, med))
    s["Recommended_Follow_Up"] = s.apply(_follow_up, axis=1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Requests scored", f"{len(s):,}")
    k2.metric("High risk", int((s["Risk"] == "High").sum()))
    k3.metric("Avg breach probability", f"{s['Breach_Probability'].mean():.1%}")
    k4.metric("Avg queue length", f"{s['Queue_Length'].mean():.1f}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(s, x="Risk", category_orders={"Risk": ["Low", "Medium", "High"]},
                                 color="Risk",
                                 color_discrete_map={"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"},
                                 title="Risk distribution"), width="stretch")
    byo = s.groupby("Assigned_Office")["Breach_Probability"].mean().mul(100).round(1).reset_index()
    c2.plotly_chart(px.bar(byo, x="Assigned_Office", y="Breach_Probability", title="Avg breach risk by office (%)"),
                    width="stretch")

    st.subheader("Review queue (highest risk first)")
    show = [c for c in [C.ID_COL, "Service_Type", "Assigned_Office", "Priority", "Queue_Length",
                        "Required_Documents_Complete", "Breach_Probability", "Risk", "Recommended_Follow_Up"]
            if c in s.columns]
    queue = s[s["Risk"] != "Low"].sort_values("Breach_Probability", ascending=False)
    st.dataframe(queue[show].head(50).style.format({"Breach_Probability": "{:.1%}"}), width="stretch")
    st.download_button("Download review queue (CSV)", queue[show].to_csv(index=False), "review_queue.csv")

    st.subheader("Record staff decision")
    st.caption("Only Student Services Staff / Office Heads may record a final decision. "
               "The model never decides automatically.")
    top = queue.head(50)
    ids = top[C.ID_COL].tolist() if C.ID_COL in top.columns else top.index.tolist()
    if not ids:
        st.success("No medium/high-risk requests at current thresholds.")
    else:
        with st.form("decision"):
            rid = st.selectbox("Request", ids)
            who = st.text_input("Reviewer name")
            dec = st.selectbox("Decision", ["Prioritize now", "Reassign staff",
                                            "Request documents from student",
                                            "No action needed (override model)"])
            note = st.text_area("Notes")
            ok = st.form_submit_button("Save decision")
        if ok:
            if not who.strip():
                st.error("Reviewer name is required.")
            else:
                st.session_state.setdefault("decisions", []).append(
                    {"Request": rid, "Reviewer": who, "Decision": dec, "Notes": note,
                     "Recorded": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "Model_Version": meta["version"]})
                st.success("Decision recorded.")
    if st.session_state.get("decisions"):
        d = pd.DataFrame(st.session_state["decisions"])
        st.dataframe(d, width="stretch")
        st.download_button("Download decision log", d.to_csv(index=False), "decision_log.csv")

    st.info("Interpretation: High risk = likely to exceed its SLA unless staff act early. "
            "Probabilities support prioritization; they are not verdicts on students or requests.")
