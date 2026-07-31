"""Central configuration: paths, feature schema, thresholds, and color constants."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURE_COLUMNS = CATEGORICAL + NUMERIC
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost"]

# Risk tiers / classification thresholds
CLASSIFICATION_THRESHOLD = 0.5
RISK_HIGH_THRESHOLD = 0.6
RISK_MEDIUM_THRESHOLD = 0.3

# Colors
COLOR_CHURN_YES = "#ef4444"
COLOR_CHURN_NO = "#8b5cf6"
COLOR_PURPLE = "#8b5cf6"
COLOR_PURPLE_DARK = "#6d28d9"
COLOR_BLUE = "#2563eb"
COLOR_GREEN = "#10b981"
COLOR_AMBER = "#f59e0b"
COLOR_RED = "#ef4444"

CHURN_COLOR_MAP = {"Yes": COLOR_CHURN_YES, "No": COLOR_CHURN_NO}
RISK_TIER_COLOR_MAP = {"High": COLOR_RED, "Medium": COLOR_AMBER, "Low": COLOR_GREEN}

PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white")
