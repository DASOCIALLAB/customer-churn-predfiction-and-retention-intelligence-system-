# Customer Churn Prediction & Retention Intelligence System

A Streamlit app that trains churn-prediction models on Telco customer data, explains
individual predictions, and simulates the ROI of retention campaigns.

## Features

- **Overview** — dataset KPIs and a leaderboard comparing Logistic Regression, Random
  Forest, and XGBoost (ROC-AUC, accuracy, precision, recall, F1).
- **EDA Explorer** — churn breakdowns by tenure, charges, services, and demographics,
  plus a correlation heatmap.
- **Predict Churn** — score a single customer via a form, see a risk gauge, top churn
  drivers (SHAP / coefficients / feature importance depending on model), and tailored
  retention recommendations.
- **Batch Analysis** — upload a CSV to score many customers at once and download the
  results.
- **ROI Simulator** — adjust risk threshold, campaign cost, and expected retention
  success rate to see projected campaign economics.

Data comes from the [Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
via `kagglehub` when available, falling back to a bundled synthetic sample otherwise.

## Project structure

```
churn-app/
├── app.py                  # Routing only: sidebar nav + calls page render functions
├── config.py               # Feature schema, thresholds, color constants, paths
├── modules/
│   ├── data_integration.py   # load_data, clean_data, build_preprocessing_pipeline
│   ├── prediction_engine.py  # train_models (cached), predict_churn, get_best_model
│   ├── retention_engine.py   # get_recommendations, get_churn_drivers, risk_tier
│   └── roi_simulator.py      # calculate_roi
├── views/
│   ├── overview.py
│   ├── eda_explorer.py
│   ├── predict_churn.py
│   ├── batch_analysis.py
│   └── roi_simulator_page.py
├── utils/
│   └── styling.py           # Custom CSS, kpi_card() helper
├── models/                  # Trained model artifacts (gitignored)
├── data/                    # Dataset CSVs (gitignored)
└── requirements.txt
```

> Note: pages live in `views/`, not `pages/` — Streamlit auto-detects a sibling
> `pages/` folder and switches into its legacy multipage-app navigation, which would
> bypass `app.py`'s own routing and add UI that isn't part of this app.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app trains all three models on first load and caches them with
`@st.cache_resource`, so switching between pages doesn't retrain anything.
