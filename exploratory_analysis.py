import pandas as pd
import plotly.express as px
import streamlit as st
import config as C


def render(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")
    st.subheader("Descriptive statistics")
    st.dataframe(df[C.NUMERIC].describe().T.round(2), width="stretch")

    has_target = C.TARGET in df.columns and df[C.TARGET].notna().any()
    if has_target:
        df = df[df[C.TARGET].notna()].copy()
        st.caption(f"Analysis uses {len(df)} completed requests with a known SLA outcome.")
    if has_target:
        st.subheader("Class distribution")
        dist = df[C.TARGET].map({0: "Within SLA", 1: "SLA Breach"}).value_counts().reset_index()
        dist.columns = ["Class", "Count"]
        c1, c2 = st.columns([1, 2])
        c1.dataframe(dist, hide_index=True)
        c2.plotly_chart(px.pie(dist, names="Class", values="Count", hole=0.4), width="stretch")
        st.caption("Imbalance matters: accuracy alone can hide poor breach detection, so recall/F1 are also reported.")
    else:
        st.info("No target column in this data, so class-based charts are skipped.")

    st.subheader("Interactive charts")
    t1, t2, t3, t4 = st.tabs(["Breach rate by category", "Numeric distribution", "Queue length vs outcome", "Correlation"])

    with t1:
        cat = st.selectbox("Group by", C.CATEGORICAL, key="eda_cat")
        if has_target:
            g = df.groupby(cat)[C.TARGET].mean().mul(100).round(1).reset_index(name="Breach rate (%)")
            st.plotly_chart(px.bar(g, x=cat, y="Breach rate (%)", color="Breach rate (%)",
                                   color_continuous_scale="Reds"), width="stretch")
        else:
            st.plotly_chart(px.histogram(df, x=cat), width="stretch")

    with t2:
        num = st.selectbox("Numeric feature", C.NUMERIC, key="eda_num")
        color = df[C.TARGET].astype(str) if has_target else None
        st.plotly_chart(px.histogram(df, x=num, nbins=30, color=color, barmode="overlay", opacity=0.7),
                        width="stretch")

    with t3:
        d = df.copy()
        if has_target:
            d[C.TARGET] = d[C.TARGET].astype(str)
        st.plotly_chart(px.box(d, x="Priority", y="Queue_Length", color=C.TARGET if has_target else None),
                        width="stretch")

    with t4:
        cols = C.NUMERIC + ([C.TARGET] if has_target else [])
        st.plotly_chart(px.imshow(df[cols].corr().round(2), text_auto=True,
                                  color_continuous_scale="RdBu_r", zmin=-1, zmax=1), width="stretch")
