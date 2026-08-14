"""Data loading and feature engineering."""

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.constants import DATA_FILENAME, SERVICE_COLUMNS, TENURE_BINS, TENURE_LABELS


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(base_dir: str | None = None) -> str:
    root = base_dir or project_root()
    return os.path.join(root, "data", DATA_FILENAME)


def load_raw_data(base_dir: str | None = None) -> pd.DataFrame:
    return pd.read_csv(get_data_path(base_dir))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Tenure_Group and Total_Services without encoding."""
    out = df.copy()
    out["Tenure_Group"] = pd.cut(
        out["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
    )
    for col in SERVICE_COLUMNS:
        out[f"{col}_Flag"] = out[col].apply(lambda x: 1 if x == "Yes" else 0)
    out["Total_Services"] = out[[f"{c}_Flag" for c in SERVICE_COLUMNS]].sum(axis=1)
    out.drop(columns=[f"{c}_Flag" for c in SERVICE_COLUMNS], inplace=True)
    return out


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out = out.dropna(subset=["TotalCharges"])
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    return out


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Clean, engineer features, and fit LabelEncoders on categoricals."""
    out = clean_raw_data(df)
    out = engineer_features(out)

    cat_cols = out.select_dtypes(include=["object", "category"]).columns
    encoders: Dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        out[col] = le.fit_transform(out[col].astype(str))
        encoders[col] = le

    return out, encoders


def prepare_raw_row(raw_input: dict) -> pd.DataFrame:
    """Build a single raw feature row from user input dict."""
    return pd.DataFrame([raw_input])


def encode_features(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    feature_names: list,
) -> pd.DataFrame:
    """Engineer + encode a raw dataframe for prediction."""
    row = engineer_features(df.copy())

    for col, le in encoders.items():
        if col not in row.columns:
            continue

        def safe_transform(val):
            val_str = str(val)
            if val_str in le.classes_:
                return le.transform([val_str])[0]
            return le.transform([le.classes_[0]])[0]

        row[col] = row[col].apply(safe_transform)

    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    return row[feature_names]
