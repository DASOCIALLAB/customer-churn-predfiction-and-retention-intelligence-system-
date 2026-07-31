"""Data loading, cleaning, and preprocessing pipeline construction."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CATEGORICAL, NUMERIC


def _generate_synthetic_telco(n=7043, seed=42):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n)
    senior = rng.choice([0, 1], n, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    tenure = rng.integers(0, 73, n)
    phone_service = rng.choice(["Yes", "No"], n, p=[0.9, 0.1])
    multiple_lines = np.where(
        phone_service == "No", "No phone service", rng.choice(["Yes", "No"], n, p=[0.42, 0.58])
    )
    internet_service = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])

    def svc_col(p_yes):
        col = rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes])
        return np.where(internet_service == "No", "No internet service", col)

    online_security = svc_col(0.29)
    online_backup = svc_col(0.34)
    device_protection = svc_col(0.34)
    tech_support = svc_col(0.29)
    streaming_tv = svc_col(0.38)
    streaming_movies = svc_col(0.39)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    paperless = rng.choice(["Yes", "No"], n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        n, p=[0.34, 0.23, 0.22, 0.21],
    )
    base_charge = rng.normal(65, 20, n).clip(18, 120)
    monthly_charges = base_charge
    monthly_charges = monthly_charges + np.where(internet_service == "Fiber optic", 20, 0)
    monthly_charges = monthly_charges + np.where(internet_service == "DSL", 5, 0)
    monthly_charges = monthly_charges + np.where(streaming_tv == "Yes", 10, 0)
    monthly_charges = monthly_charges + np.where(streaming_movies == "Yes", 10, 0)
    monthly_charges = monthly_charges.clip(18, 120)
    total_charges = (monthly_charges * tenure) + rng.normal(0, 50, n)
    total_charges = total_charges.clip(0, None)

    z = (
        -1.6
        + 1.4 * (contract == "Month-to-month")
        - 1.0 * (contract == "Two year")
        - 0.02 * tenure
        + 0.015 * (monthly_charges - 65)
        + 0.5 * (internet_service == "Fiber optic")
        + 0.4 * (online_security == "No")
        + 0.4 * (tech_support == "No")
        + 0.3 * (paperless == "Yes")
        + 0.4 * (payment == "Electronic check")
        - 0.3 * (partner == "Yes")
        - 0.2 * (dependents == "Yes")
        + 0.15 * senior
    )
    prob = 1 / (1 + np.exp(-z))
    churn = rng.binomial(1, prob)
    churn_label = np.where(churn == 1, "Yes", "No")

    return pd.DataFrame(
        {
            "customerID": [f"CUST-{i:05d}" for i in range(n)],
            "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
            "InternetService": internet_service, "OnlineSecurity": online_security,
            "OnlineBackup": online_backup, "DeviceProtection": device_protection,
            "TechSupport": tech_support, "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
            "Contract": contract, "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges.round(2), "TotalCharges": total_charges.round(2),
            "Churn": churn_label,
        }
    )


def clean_data(df):
    df = df.copy()
    if "customerID" not in df.columns:
        df.insert(0, "customerID", [f"CUST-{i:05d}" for i in range(len(df))])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"]).fillna(0)
    df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    df["Churn"] = df["Churn"].astype(str).str.strip()
    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Loading Telco churn dataset...")
def load_data():
    try:
        import glob

        import kagglehub

        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        csv_files = glob.glob(f"{path}/*.csv")
        if not csv_files:
            raise FileNotFoundError("No CSV found in kagglehub download")
        df = pd.read_csv(csv_files[0])
        source = "Kaggle (blastchar/telco-customer-churn) via kagglehub"
    except Exception:
        df = _generate_synthetic_telco()
        source = "Bundled synthetic sample (Kaggle download unavailable)"
    return clean_data(df), source


def build_preprocessing_pipeline():
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
