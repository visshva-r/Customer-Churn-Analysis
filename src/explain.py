"""SHAP explainability and plain-English summaries."""

from typing import Dict, List, Tuple

import pandas as pd

from src.labels import format_shap_series

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _risk_rules(raw_input: dict) -> List[str]:
    factors = []
    if raw_input.get("Contract") == "Month-to-month":
        factors.append("month-to-month contract")
    if raw_input.get("tenure", 99) <= 12:
        factors.append(f"short tenure ({int(raw_input.get('tenure', 0))} months)")
    if raw_input.get("PaymentMethod") == "Electronic check":
        factors.append("electronic check billing")
    if raw_input.get("InternetService") == "Fiber optic":
        factors.append("fiber internet plan")
    if raw_input.get("TechSupport") == "No" and raw_input.get("InternetService") != "No":
        factors.append("no tech support")
    if raw_input.get("OnlineSecurity") == "No" and raw_input.get("InternetService") != "No":
        factors.append("no online security")
    return factors[:5]


def explain_prediction(
    model,
    scaler,
    features: pd.DataFrame,
    feature_names: list,
    raw_input: dict,
    top_n: int = 5,
) -> Tuple[pd.Series, str]:
    scaled = scaler.transform(features)

    if SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaled)
            if isinstance(shap_values, list):
                values = shap_values[1][0]
            else:
                values = shap_values[0]
            contrib = pd.Series(values, index=feature_names).sort_values(
                key=lambda s: s.abs(), ascending=False
            )
            top = format_shap_series(contrib.head(top_n), raw_input)
            labels = [idx for idx in top.index]
            summary = "Main drivers: " + "; ".join(labels[:3]) + "."
            return top, summary
        except Exception:
            pass

    factors = _risk_rules(raw_input)
    if factors:
        return pd.Series(dtype=float), "Risk signals: " + ", ".join(factors) + "."
    return pd.Series(dtype=float), "Balanced profile across contract, tenure, and services."


def retention_recommendations(raw_input: dict, predicted_churn: str, proba: float) -> List[str]:
    if predicted_churn != "Yes":
        return [
            "Maintain current plan — low immediate churn signal.",
            "Optional: offer a loyalty perk before contract renewal.",
        ]

    recs = []
    if raw_input.get("Contract") == "Month-to-month":
        recs.append("Offer a 12-month contract with a one-time bill credit.")
    if raw_input.get("tenure", 99) <= 12:
        recs.append("Assign a onboarding specialist for the first 90 days.")
    if raw_input.get("PaymentMethod") == "Electronic check":
        recs.append("Move to auto-pay with a $10/month discount for 6 months.")
    if raw_input.get("TechSupport") == "No" and raw_input.get("InternetService") != "No":
        recs.append("Include tech support free for 3 months.")
    if proba >= 0.7:
        recs.append("Priority retention queue — manager callback within 48 hours.")
    if not recs:
        recs.append("Personalized plan review with a retention specialist.")
    return recs[:3]
