"""Business impact metrics and messaging."""

from typing import Dict

import numpy as np
import pandas as pd


def threshold_summary(threshold_curve: pd.DataFrame, threshold: float) -> Dict[str, float]:
    """Interpolate business metrics at the selected threshold."""
    curve = threshold_curve.sort_values("threshold")
    recall = float(
        np.interp(threshold, curve["threshold"], curve["recall_churn"])
    )
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
    }


def business_message(threshold: float, summary: Dict[str, float]) -> str:
    return (
        f"At threshold **{threshold:.2f}**, the model catches "
        f"**{summary['recall_churn_pct']:.1f}%** of churners but must contact "
        f"**{summary['flagged_pct']:.1f}%** of all customers "
        f"(churn precision: **{summary['precision_churn_pct']:.1f}%**)."
    )
