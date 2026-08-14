"""SHAP explainability and plain-English summaries."""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def _risk_rules(raw_input: dict) -> List[str]:
    """Rule-based risk factors from raw customer fields."""
    factors = []
    if raw_input.get("Contract") == "Month-to-month":
        factors.append("month-to-month contract")
    if raw_input.get("tenure", 99) <= 12:
        factors.append("short tenure (≤12 months)")
    if raw_input.get("PaymentMethod") == "Electronic check":
        factors.append("electronic check payment")
    if raw_input.get("InternetService") == "Fiber optic":
        factors.append("fiber optic internet (higher churn segment)")
    if raw_input.get("TechSupport") == "No" and raw_input.get("InternetService") != "No":
        factors.append("no tech support on internet plan")
    if raw_input.get("OnlineSecurity") == "No" and raw_input.get("InternetService") != "No":
        factors.append("no online security add-on")
    return factors[:5]


def explain_prediction(
    model,
    scaler,
    features: pd.DataFrame,
    feature_names: list,
    raw_input: dict,
    top_n: int = 5,
) -> Tuple[pd.Series, str]:
    """Return top SHAP contributors and a plain-English summary."""
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
            top = contrib.head(top_n)
            top_labels = ", ".join(top.index.tolist())
            summary = f"Higher risk mainly driven by model features: {top_labels}."
        except Exception:
            top = pd.Series(dtype=float)
            summary = _plain_summary(raw_input)
    else:
        top = pd.Series(dtype=float)
        summary = _plain_summary(raw_input)

    if not top.empty:
        return top, summary
    return top, _plain_summary(raw_input)


def _plain_summary(raw_input: dict) -> str:
    factors = _risk_rules(raw_input)
    if factors:
        return "Higher risk mainly due to: " + ", ".join(factors) + "."
    return "Risk profile is moderate based on contract, tenure, and service usage."


def retention_recommendations(raw_input: dict, predicted_churn: str) -> List[str]:
    """Rule-based retention actions for high-risk customers."""
    if predicted_churn != "Yes":
        return [
            "Customer appears stable — continue standard engagement.",
            "Consider upselling value-add services (security, backup, streaming bundles).",
        ]

    recs = []
    if raw_input.get("Contract") == "Month-to-month":
        recs.append("Offer a discounted 1-year contract to reduce churn risk.")
    if raw_input.get("tenure", 99) <= 12:
        recs.append("Send a welcome/loyalty outreach in the first 90 days.")
    if raw_input.get("PaymentMethod") == "Electronic check":
        recs.append("Switch customer to automatic bank transfer with a small incentive.")
    if raw_input.get("TechSupport") == "No":
        recs.append("Bundle tech support or run a proactive support check-in.")
    if not recs:
        recs.append("Offer a targeted retention discount or personalized plan review.")
    recs.append("Schedule a retention call within 7 days for at-risk accounts.")
    return recs[:3]
