"""Overview page: KPIs, model leaderboard, and high-level churn charts."""

import pandas as pd
import plotly.express as px
import streamlit as st

from config import PLOT_LAYOUT
from utils.styling import kpi_card


def render(df, bundle):
    st.title("📉 Customer Churn Prediction & Retention Intelligence System")
    st.caption(f"Data source: {st.session_state['data_source']}  |  {len(df):,} customers")

    churn_rate = (df["Churn"] == "Yes").mean() * 100
    avg_tenure = df["tenure"].mean()
    avg_charges = df["MonthlyCharges"].mean()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Total Customers", f"{len(df):,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Churn Rate", f"{churn_rate:.1f}%"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Avg Tenure", f"{avg_tenure:.1f} mo"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Avg Monthly Charges", f"${avg_charges:.2f}"), unsafe_allow_html=True)

    st.markdown("### Model Performance")
    results = bundle["results"]
    best_name = bundle["best_name"]
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    res_df = res_df.sort_values("roc_auc", ascending=False)
    st.success(
        f"🏆 Best Model: **{best_name}** (ROC-AUC = {results[best_name]['roc_auc']:.3f}) "
        "— used across Predict, Batch, and ROI pages."
    )
    st.dataframe(
        res_df.style.format({"roc_auc": "{:.3f}", "accuracy": "{:.3f}", "precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
        width='stretch',
        hide_index=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        churn_counts = df["Churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn", "Count"]
        fig = px.pie(
            churn_counts, names="Churn", values="Count", hole=0.6, color="Churn",
            color_discrete_map={"Yes": "#ef4444", "No": "#8b5cf6"}, title="Churn Distribution",
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        by_contract = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        by_contract.columns = ["Contract", "Churn Rate (%)"]
        fig2 = px.bar(
            by_contract, x="Contract", y="Churn Rate (%)", color="Contract",
            color_discrete_sequence=["#6d28d9", "#2563eb", "#8b5cf6"], title="Churn Rate by Contract Type",
        )
        fig2.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
