"""EDA Explorer page: exploratory charts across tenure/charges, services, demographics, correlations."""

import plotly.express as px
import streamlit as st

from config import PLOT_LAYOUT


def render(df):
    st.title("🔍 EDA Explorer")
    tabs = st.tabs(["Tenure & Charges", "Services", "Demographics", "Correlations"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                df, x="tenure", color="Churn", barmode="overlay", nbins=30,
                color_discrete_map={"Yes": "#ef4444", "No": "#8b5cf6"}, title="Tenure Distribution by Churn",
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.histogram(
                df, x="MonthlyCharges", color="Churn", barmode="overlay", nbins=30,
                color_discrete_map={"Yes": "#ef4444", "No": "#8b5cf6"}, title="Monthly Charges by Churn",
            )
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
        fig = px.box(
            df, x="Churn", y="TotalCharges", color="Churn",
            color_discrete_map={"Yes": "#ef4444", "No": "#8b5cf6"}, title="Total Charges by Churn",
        )
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        service_cols = [
            "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines", "PhoneService",
        ]
        chosen = st.selectbox("Service feature", service_cols)
        rate = df.groupby(chosen)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        rate.columns = [chosen, "Churn Rate (%)"]
        fig = px.bar(rate, x=chosen, y="Churn Rate (%)", color=chosen, title=f"Churn Rate by {chosen}")
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        demo_cols = ["gender", "SeniorCitizen", "Partner", "Dependents"]
        c1, c2 = st.columns(2)
        for i, col in enumerate(demo_cols):
            rate = df.groupby(col)["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            rate.columns = [col, "Churn Rate (%)"]
            fig = px.bar(rate, x=col, y="Churn Rate (%)", title=f"Churn Rate by {col}", color=col)
            fig.update_layout(**PLOT_LAYOUT, showlegend=False)
            (c1 if i % 2 == 0 else c2).plotly_chart(fig, width='stretch')

    with tabs[3]:
        numeric_df = df[["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]].copy()
        numeric_df["Churn"] = (df["Churn"] == "Yes").astype(int)
        corr = numeric_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Purples", title="Correlation Heatmap")
        fig.update_layout(**PLOT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
