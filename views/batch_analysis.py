"""Batch Analysis page: CSV upload, bulk scoring, and results export."""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import FEATURE_COLUMNS, PLOT_LAYOUT
from modules.prediction_engine import predict_churn
from modules.retention_engine import risk_tier
from utils.styling import kpi_card


def render(df, bundle):
    st.title("📊 Batch Analysis")
    pipeline = bundle["pipelines"][bundle["best_name"]]
    st.write("Upload a CSV with the standard Telco churn columns to score multiple customers at once.")
    st.download_button(
        "Download sample template",
        df.drop(columns=["Churn"]).head(20).to_csv(index=False),
        file_name="churn_batch_template.csv",
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        batch_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return

    missing = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")
        return

    work = batch_df.copy()
    work["TotalCharges"] = pd.to_numeric(work["TotalCharges"], errors="coerce")
    work["TotalCharges"] = work["TotalCharges"].fillna(work["tenure"] * work["MonthlyCharges"]).fillna(0)
    work["SeniorCitizen"] = pd.to_numeric(work["SeniorCitizen"], errors="coerce").fillna(0).astype(int)

    probs = predict_churn(pipeline, work[FEATURE_COLUMNS])
    work["Churn Probability"] = (probs * 100).round(1)
    work["Risk Tier"] = [risk_tier(p)[0] for p in probs]

    st.success(f"Scored {len(work):,} customers.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi_card("Customers Scored", f"{len(work):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("High Risk", f"{(work['Risk Tier'] == 'High').sum():,}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Churn Prob", f"{work['Churn Probability'].mean():.1f}%"), unsafe_allow_html=True)

    display_cols = (["customerID"] if "customerID" in work.columns else []) + ["Churn Probability", "Risk Tier"] + FEATURE_COLUMNS
    st.dataframe(work[display_cols].sort_values("Churn Probability", ascending=False), width='stretch', hide_index=True)

    csv_bytes = work.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv_bytes, file_name="churn_predictions.csv", mime="text/csv")

    c1, c2 = st.columns(2)
    with c1:
        tier_counts = work["Risk Tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        fig = px.pie(
            tier_counts, names="Risk Tier", values="Count", hole=0.5, color="Risk Tier",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            title="Risk Tier Distribution",
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(work, x="Churn Probability", nbins=30, color_discrete_sequence=["#8b5cf6"], title="Churn Probability Distribution")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
