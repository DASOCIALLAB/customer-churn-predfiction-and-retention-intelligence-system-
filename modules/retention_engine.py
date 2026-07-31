"""Churn risk tiering, driver explainability, and retention recommendations."""

import numpy as np

from config import CATEGORICAL, RISK_HIGH_THRESHOLD, RISK_MEDIUM_THRESHOLD

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def clean_feature_name(name):
    name = name.replace("cat__", "").replace("num__", "")
    for col in CATEGORICAL:
        prefix = col + "_"
        if name.startswith(prefix):
            return f"{col} = {name[len(prefix):]}"
    return name


def get_churn_drivers(pipeline, model_name, X_instance_df, top_n=5):
    preprocess = pipeline.named_steps["preprocess"]
    clf = pipeline.named_steps["clf"]
    X_trans = preprocess.transform(X_instance_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    feature_names = preprocess.get_feature_names_out()

    contributions = None
    method = "coefficient magnitude"

    if model_name == "Logistic Regression":
        contributions = clf.coef_[0] * X_trans[0]
        method = "coefficient magnitude"
    elif SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(clf)
            sv = np.array(explainer.shap_values(X_trans))
            if sv.ndim == 3:
                contributions = sv[0, :, -1] if sv.shape[0] == X_trans.shape[0] else sv[-1][0]
            elif sv.ndim == 2:
                contributions = sv[0]
            else:
                contributions = sv
            method = "SHAP"
        except Exception:
            contributions = None

    if contributions is None:
        contributions = clf.feature_importances_ * X_trans[0]
        method = "feature importance"

    contributions = np.asarray(contributions).flatten()
    order = np.argsort(-np.abs(contributions))[:top_n]
    drivers = [
        {"feature": clean_feature_name(feature_names[i]), "impact": float(contributions[i])}
        for i in order
    ]
    return drivers, method


def risk_tier(prob):
    if prob >= RISK_HIGH_THRESHOLD:
        return "High", "risk-badge-high"
    if prob >= RISK_MEDIUM_THRESHOLD:
        return "Medium", "risk-badge-medium"
    return "Low", "risk-badge-low"


def get_recommendations(row, prob):
    recs = []
    if prob >= RISK_HIGH_THRESHOLD:
        recs.append(("🎯 Retention Offer", "High churn risk — offer a loyalty discount or sign-up incentive (e.g. 15-20% off for switching to an annual contract)."))
    if row["Contract"] == "Month-to-month":
        recs.append(("📄 Contract Upgrade", "Customer is on month-to-month billing — propose a 1- or 2-year contract with a bundled discount to increase lock-in."))
    if row["TechSupport"] == "No" and row["InternetService"] != "No":
        recs.append(("🛠️ Tech Support Offer", "No tech support subscribed — offer a free trial of premium tech support to reduce service frustration."))
    if row["OnlineSecurity"] == "No" and row["InternetService"] != "No":
        recs.append(("🔒 Security Bundle", "No online security add-on — bundle security services to increase perceived value."))
    if row["MonthlyCharges"] > 80:
        recs.append(("💳 Plan Optimization", "High monthly bill — review plan for right-sizing or offer a discounted bundle to ease price sensitivity."))
    if row["tenure"] < 12:
        recs.append(("🤝 Engagement Outreach", "New/low-tenure customer — proactive onboarding check-in to boost engagement in the first year."))
    if row["PaymentMethod"] == "Electronic check":
        recs.append(("💵 Payment Method Nudge", "Electronic check users churn more often — incentivize switching to autopay (bank transfer/credit card) with a small credit."))
    if not recs:
        recs.append(("✅ Maintain Engagement", "Low risk customer — continue standard engagement and satisfaction surveys."))
    return recs
