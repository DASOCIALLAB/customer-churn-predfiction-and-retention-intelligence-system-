"""Customer Churn Prediction & Retention Intelligence System."""

import streamlit as st

from modules.data_integration import load_data
from modules.prediction_engine import train_models
from utils.styling import inject_css
from views import batch_analysis, eda_explorer, overview, predict_churn, roi_simulator_page

st.set_page_config(
    page_title="Customer Churn Prediction & Retention Intelligence System",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    inject_css()
    df, source = load_data()
    st.session_state["data_source"] = source
    bundle = train_models(df)

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Overview", "EDA Explorer", "Predict Churn", "Batch Analysis", "ROI Simulator"],
        label_visibility="collapsed",
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"Data source:\n\n{source}")
    st.sidebar.caption(f"Best model: **{bundle['best_name']}**")

    if page == "Overview":
        overview.render(df, bundle)
    elif page == "EDA Explorer":
        eda_explorer.render(df)
    elif page == "Predict Churn":
        predict_churn.render(df, bundle)
    elif page == "Batch Analysis":
        batch_analysis.render(df, bundle)
    elif page == "ROI Simulator":
        roi_simulator_page.render(df, bundle)


if __name__ == "__main__":
    main()
