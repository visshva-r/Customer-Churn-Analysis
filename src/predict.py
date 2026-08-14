"""Single and batch prediction helpers."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.data import clean_raw_data, encode_features


def predict_proba(
    model,
    scaler,
    features: pd.DataFrame,
) -> np.ndarray:
    scaled = scaler.transform(features)
    return model.predict_proba(scaled)[:, 1]


def predict_single(
    model,
    scaler,
    encoders: Dict,
    feature_names: list,
    raw_input: dict,
    threshold: float = 0.5,
) -> Tuple[float, str, pd.DataFrame]:
    features = encode_features(
        pd.DataFrame([raw_input]), encoders, feature_names
    )
    proba = predict_proba(model, scaler, features)[0]
    label = "Yes" if proba >= threshold else "No"
    return proba, label, features


def predict_batch(
    model,
    scaler,
    encoders: Dict,
    feature_names: list,
    raw_df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Predict churn for a batch of raw customer records."""
    df = clean_raw_data(raw_df.copy())
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    features = encode_features(df, encoders, feature_names)
    probas = predict_proba(model, scaler, features)

    out = df.copy()
    out["Churn_Probability"] = probas
    out["Churn_Prediction"] = np.where(probas >= threshold, "Yes", "No")
    return out
