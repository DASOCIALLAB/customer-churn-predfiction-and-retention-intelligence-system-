"""Model training, caching, and churn probability prediction."""

import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from config import CLASSIFICATION_THRESHOLD, FEATURE_COLUMNS, MODEL_NAMES
from modules.data_integration import build_preprocessing_pipeline


def make_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42, eval_metric="logloss", n_jobs=-1,
        )
    raise ValueError(name)


def get_best_model(results):
    return max(results, key=lambda k: results[k]["roc_auc"])


@st.cache_resource(show_spinner="Training churn prediction models (Logistic Regression, Random Forest, XGBoost)...")
def train_models(df):
    X = df[FEATURE_COLUMNS]
    y = (df["Churn"] == "Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}
    pipelines = {}
    for name in MODEL_NAMES:
        eval_pipe = Pipeline([("preprocess", build_preprocessing_pipeline()), ("clf", make_model(name))])
        eval_pipe.fit(X_train, y_train)
        proba = eval_pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= CLASSIFICATION_THRESHOLD).astype(int)
        results[name] = {
            "roc_auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        full_pipe = Pipeline([("preprocess", build_preprocessing_pipeline()), ("clf", make_model(name))])
        full_pipe.fit(X, y)
        pipelines[name] = full_pipe

    best_name = get_best_model(results)
    return {"results": results, "pipelines": pipelines, "best_name": best_name}


def predict_churn(pipeline, X_df):
    return pipeline.predict_proba(X_df)[:, 1]


@st.cache_data(show_spinner="Scoring all customers...")
def score_all(_pipeline, df, feature_cols):
    return _pipeline.predict_proba(df[feature_cols])[:, 1]
