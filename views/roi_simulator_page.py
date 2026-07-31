"""ROI Simulator page: retention campaign economics under adjustable assumptions."""

import plotly.graph_objects as go
import streamlit as st

from config import FEATURE_COLUMNS, PLOT_LAYOUT
from modules.prediction_engine import score_all
from modules.roi_simulator import calculate_roi
from utils.styling import kpi_card


def render(df, bundle):
    st.title("💰 ROI Simulator")
    pipeline = bundle["pipelines"][bundle["best_name"]]
    probs = score_all(pipeline, df, FEATURE_COLUMNS)
    work = df.copy()
    work["churn_prob"] = probs

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.slider("Risk Threshold", 0.0, 1.0, 0.5, 0.01)
    with c2:
        cost_per_customer = st.slider("Cost per Targeted Customer ($)", 5, 200, 50, 5)
    with c3:
        success_rate = st.slider("Expected Retention Success Rate", 0.0, 1.0, 0.3, 0.05)

    roi = calculate_roi(work, threshold, cost_per_customer, success_rate)
    n_targeted = roi["n_targeted"]
    campaign_cost = roi["campaign_cost"]
    monthly_revenue_saved = roi["monthly_revenue_saved"]
    net_roi = roi["net_roi"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi_card("Customers Targeted", f"{n_targeted:,}"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi_card("Campaign Cost", f"${campaign_cost:,.0f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi_card("Monthly Revenue Saved", f"${monthly_revenue_saved:,.0f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi_card("Net ROI", f"${net_roi:,.0f}"), unsafe_allow_html=True)

    fig = go.Figure(
        go.Bar(
            x=["Campaign Cost", "Monthly Revenue Saved", "Net ROI"],
            y=[campaign_cost, monthly_revenue_saved, net_roi],
            marker_color=["#ef4444", "#8b5cf6", "#10b981" if net_roi >= 0 else "#ef4444"],
        )
    )
    fig.update_layout(**PLOT_LAYOUT, title="Campaign Economics")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"At a {threshold:.0%} risk threshold, {n_targeted:,} of {len(work):,} customers "
        f"({n_targeted / len(work) * 100:.1f}%) are targeted for retention outreach."
    )
