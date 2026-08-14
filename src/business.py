"""Business impact metrics and messaging."""

from typing import Dict

import numpy as np
import pandas as pd

# Assumed average revenue at risk per churner (illustrative for retention ROI)
DEFAULT_LTV_USD = 500
DEFAULT_CHURN_RATE = 0.27


def threshold_summary(threshold_curve: pd.DataFrame, threshold: float) -> Dict[str, float]:
    curve = threshold_curve.sort_values("threshold")
    recall = float(np.interp(threshold, curve["threshold"], curve["recall_churn"]))
    precision = float(
        np.interp(threshold, curve["threshold"], curve["precision_churn"])
    )
    flagged_pct = float(
        np.interp(threshold, curve["threshold"], curve["flagged_pct"])
    )
    return {
        "recall_churn_pct": recall * 100,
        "precision_churn_pct": precision * 100,
        "flagged_pct": flagged_pct * 100,
        "recall_churn": recall,
    }


def business_message(threshold: float, summary: Dict[str, float]) -> str:
    return (
        f"At cutoff **{threshold:.2f}**, the model finds "
        f"**{summary['recall_churn_pct']:.1f}%** of churners and flags "
        f"**{summary['flagged_pct']:.1f}%** of customers "
        f"(precision **{summary['precision_churn_pct']:.1f}%**)."
    )


def retention_roi_estimate(
    summary: Dict[str, float],
    portfolio_size: int = 1000,
    ltv_usd: float = DEFAULT_LTV_USD,
    churn_rate: float = DEFAULT_CHURN_RATE,
) -> Dict[str, float]:
    """Rough ROI if retention saves a share of flagged churners."""
    expected_churners = portfolio_size * churn_rate
    recall = summary.get("recall_churn", summary["recall_churn_pct"] / 100)
    churners_caught = expected_churners * recall
    customers_flagged = portfolio_size * (summary["flagged_pct"] / 100)
    revenue_at_risk = expected_churners * ltv_usd
    revenue_protected = churners_caught * ltv_usd
    missed_churners = expected_churners - churners_caught
    missed_revenue = missed_churners * ltv_usd
    return {
        "portfolio_size": portfolio_size,
        "ltv_usd": ltv_usd,
        "expected_churners": expected_churners,
        "churners_caught": churners_caught,
        "customers_flagged": customers_flagged,
        "revenue_at_risk": revenue_at_risk,
        "revenue_protected": revenue_protected,
        "missed_revenue": missed_revenue,
    }
