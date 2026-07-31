"""Predict Churn page: single-customer form, gauge, top drivers, and recommendations."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import FEATURE_COLUMNS, PLOT_LAYOUT
from modules.prediction_engine import predict_churn
from modules.retention_engine import get_churn_drivers, get_recommendations, risk_tier


def gauge_chart(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 40, "color": "white"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "white"},
                "bar": {"color": "#8b5cf6"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "#10b981"},
                    {"range": [30, 60], "color": "#f59e0b"},
                    {"range": [60, 100], "color": "#ef4444"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": prob * 100},
            },
            title={"text": "Churn Probability", "font": {"color": "white"}},
        )
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "white"}, height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render(df, bundle):
    st.title("🎯 Predict Churn")
    pipeline = bundle["pipelines"][bundle["best_name"]]
    st.caption(f"Model in use: **{bundle['best_name']}**")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", sorted(df["gender"].unique()))
            senior_label = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", sorted(df["Partner"].unique()))
            dependents = st.selectbox("Dependents", sorted(df["Dependents"].unique()))
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with c2:
            phone_service = st.selectbox("Phone Service", sorted(df["PhoneService"].unique()))
            multiple_lines = st.selectbox("Multiple Lines", sorted(df["MultipleLines"].unique()))
            internet_service = st.selectbox("Internet Service", sorted(df["InternetService"].unique()))
            online_security = st.selectbox("Online Security", sorted(df["OnlineSecurity"].unique()))
            online_backup = st.selectbox("Online Backup", sorted(df["OnlineBackup"].unique()))
        with c3:
            device_protection = st.selectbox("Device Protection", sorted(df["DeviceProtection"].unique()))
            tech_support = st.selectbox("Tech Support", sorted(df["TechSupport"].unique()))
            streaming_tv = st.selectbox("Streaming TV", sorted(df["StreamingTV"].unique()))
            streaming_movies = st.selectbox("Streaming Movies", sorted(df["StreamingMovies"].unique()))
            contract = st.selectbox("Contract", sorted(df["Contract"].unique()))

        c4, c5, c6 = st.columns(3)
        with c4:
            paperless = st.selectbox("Paperless Billing", sorted(df["PaperlessBilling"].unique()))
        with c5:
            payment = st.selectbox("Payment Method", sorted(df["PaymentMethod"].unique()))
        with c6:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

        total_charges_input = st.number_input(
            "Total Charges ($) — leave at 0 to auto-calculate from tenure × monthly charges",
            min_value=0.0, value=0.0, step=10.0,
        )

        submitted = st.form_submit_button("Predict Churn Risk", width='stretch')

    if submitted:
        total_charges = total_charges_input if total_charges_input > 0 else tenure * monthly_charges
        instance = pd.DataFrame(
            [
                {
                    "gender": gender, "SeniorCitizen": 1 if senior_label == "Yes" else 0,
                    "Partner": partner, "Dependents": dependents, "tenure": tenure,
                    "PhoneService": phone_service, "MultipleLines": multiple_lines,
                    "InternetService": internet_service, "OnlineSecurity": online_security,
                    "OnlineBackup": online_backup, "DeviceProtection": device_protection,
                    "TechSupport": tech_support, "StreamingTV": streaming_tv,
                    "StreamingMovies": streaming_movies, "Contract": contract,
                    "PaperlessBilling": paperless, "PaymentMethod": payment,
                    "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
                }
            ]
        )[FEATURE_COLUMNS]

        prob = float(predict_churn(pipeline, instance)[0])
        tier, badge_class = risk_tier(prob)

        st.markdown("---")
        r1, r2 = st.columns([1, 1])
        with r1:
            st.plotly_chart(gauge_chart(prob), width='stretch')
        with r2:
            st.markdown(f"<div style='margin-top:40px;'><span class='{badge_class}'>{tier} Risk</span></div>", unsafe_allow_html=True)
            st.metric("Churn Probability", f"{prob * 100:.1f}%")

        st.markdown("### Top Churn Drivers")
        drivers, method = get_churn_drivers(pipeline, bundle["best_name"], instance, top_n=5)
        st.caption(f"Explanation method: {method}")
        drv_df = pd.DataFrame(drivers)
        colors = ["#ef4444" if v > 0 else "#10b981" for v in drv_df["impact"]]
        fig = go.Figure(go.Bar(x=drv_df["impact"], y=drv_df["feature"], orientation="h", marker_color=colors))
        fig.update_layout(
            **PLOT_LAYOUT, xaxis_title="Impact on churn probability", yaxis_title="",
            height=320, yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Retention Recommendations")
        row = instance.iloc[0]
        for title, desc in get_recommendations(row, prob):
            st.markdown(f"<div class='rec-card'><b>{title}</b><br>{desc}</div>", unsafe_allow_html=True)
