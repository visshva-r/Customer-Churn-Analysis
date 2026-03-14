"""
Test script for the churn app: data load, preprocessing, training, and prediction.
Run from project root: python -m pytest tests/test_app.py -v
Or: python tests/test_app.py
"""
import os
import sys

# Run from project root so app can find data/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_load_raw_data():
    """Test data loading."""
    import app
    df = app.load_raw_data()
    assert df is not None
    assert len(df) > 0
    assert "Churn" in df.columns
    assert "TotalCharges" in df.columns
    print("OK load_raw_data")


def test_preprocess_data():
    """Test preprocessing pipeline."""
    import app
    raw = app.load_raw_data()
    df, encoders = app.preprocess_data(raw)
    assert "Churn" in df.columns
    assert "Tenure_Group" in df.columns
    assert "Total_Services" in df.columns
    assert df["TotalCharges"].dtype in (float, "float64", "float32")
    assert len(encoders) > 0
    print("OK preprocess_data")


def test_train_model():
    """Test model training and metrics."""
    import app
    model, scaler, encoders, feature_names, metrics = app.train_model()
    assert model is not None
    assert scaler is not None
    assert len(feature_names) > 0
    assert "accuracy" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert "feature_importance" in metrics
    assert "threshold_curve" in metrics
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    cm = metrics["confusion_matrix"]
    assert cm.shape == (2, 2)
    fi = metrics["feature_importance"]
    assert len(fi) == len(feature_names)
    th = metrics["threshold_curve"]
    assert "recall_churn" in th.columns and "threshold" in th.columns
    print("OK train_model")


def test_prediction_path():
    """Test prediction with a sample row (same features as training)."""
    import app
    import pandas as pd
    import numpy as np
    model, scaler, encoders, feature_names, _ = app.train_model()
    raw = app.load_raw_data()
    df, _ = app.preprocess_data(raw)
    X = df.drop(columns=["Churn"])
    # Use first row as 2D array for scaler/model
    row = X.iloc[[0]]
    scaled = scaler.transform(row)
    proba = model.predict_proba(scaled)[0, 1]
    pred = (proba >= 0.5).astype(int)
    assert 0 <= proba <= 1
    assert pred in (0, 1)
    # Batch of 3
    row3 = X.iloc[:3]
    scaled3 = scaler.transform(row3)
    proba3 = model.predict_proba(scaled3)[:, 1]
    assert proba3.shape == (3,)
    print("OK prediction_path")


def test_threshold_curve_no_nan():
    """Ensure threshold curve has no NaN that could break charts."""
    import app
    _, _, _, _, metrics = app.train_model()
    th = metrics["threshold_curve"]
    assert th[["recall_churn", "precision_churn", "flagged_pct"]].notna().all().all()
    print("OK threshold_curve_no_nan")


def test_feature_importance_chart_data():
    """Feature importance for bar chart: 10 rows, no NaN."""
    import app
    _, _, _, _, metrics = app.train_model()
    fi = metrics["feature_importance"].head(10).sort_values()
    assert len(fi) <= 10
    assert fi.notna().all()
    assert (fi >= 0).all()
    print("OK feature_importance_chart_data")


if __name__ == "__main__":
    test_load_raw_data()
    test_preprocess_data()
    test_train_model()
    test_prediction_path()
    test_threshold_curve_no_nan()
    test_feature_importance_chart_data()
    print("\nAll tests passed.")
