"""
Test script for the churn pipeline.
Run from project root: python tests/test_app.py
"""
import os
import sys
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_load_raw_data():
    from src.data import load_raw_data

    df = load_raw_data()
    assert len(df) > 0
    assert "Churn" in df.columns
    print("OK load_raw_data")


def test_preprocess_data():
    from src.data import load_raw_data, preprocess_data

    raw = load_raw_data()
    df, encoders = preprocess_data(raw)
    assert "Tenure_Group" in df.columns
    assert "Total_Services" in df.columns
    assert len(encoders) > 0
    print("OK preprocess_data")


def test_train_pipeline():
    from src.train import train_pipeline

    artifacts = train_pipeline()
    metrics = artifacts["metrics"]
    assert 0.75 <= metrics["roc_auc"] <= 0.85
    assert metrics["churn_recall"] >= 0.7
    assert "baseline" in metrics and "xgb" in metrics
    print("OK train_pipeline")


def test_save_and_load_artifacts():
    from src.train import artifacts_exist, load_artifacts, save_artifacts, train_pipeline

    with tempfile.TemporaryDirectory() as tmp:
        artifacts = train_pipeline(tmp)
        save_artifacts(artifacts, tmp)
        assert artifacts_exist(tmp)
        model, scaler, encoders, feature_names, metrics = load_artifacts(tmp)
        assert model is not None
        assert len(feature_names) > 0
        assert "roc_auc" in metrics
    print("OK save_and_load_artifacts")


def test_single_and_batch_prediction():
    from src.data import load_raw_data, preprocess_data
    from src.predict import predict_batch, predict_single
    from src.train import train_pipeline

    artifacts = train_pipeline()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    encoders = artifacts["encoders"]
    feature_names = artifacts["feature_names"]

    raw = load_raw_data().drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    proba, label, _ = predict_single(
        model, scaler, encoders, feature_names, raw, threshold=0.5
    )
    assert 0 <= proba <= 1
    assert label in ("Yes", "No")

    batch = load_raw_data().head(5).drop(columns=["Churn"])
    results = predict_batch(model, scaler, encoders, feature_names, batch)
    assert "Churn_Probability" in results.columns
    assert len(results) == 5
    print("OK single_and_batch_prediction")


def test_explain_and_business():
    from src.business import business_message, threshold_summary
    from src.data import load_raw_data
    from src.explain import explain_prediction, retention_recommendations
    from src.predict import predict_single
    from src.train import train_pipeline

    artifacts = train_pipeline()
    raw = load_raw_data().drop(columns=["customerID", "Churn"]).iloc[0].to_dict()
    _, pred, features = predict_single(
        artifacts["model"],
        artifacts["scaler"],
        artifacts["encoders"],
        artifacts["feature_names"],
        raw,
    )
    top, summary = explain_prediction(
        artifacts["model"],
        artifacts["scaler"],
        features,
        artifacts["feature_names"],
        raw,
    )
    assert isinstance(summary, str)
    recs = retention_recommendations(raw, pred, 0.5)
    assert len(recs) >= 2
    th_summary = threshold_summary(artifacts["metrics"]["threshold_curve"], 0.5)
    msg = business_message(0.5, th_summary)
    assert "threshold" in msg.lower() or "0.50" in msg
    print("OK explain_and_business")


if __name__ == "__main__":
    test_load_raw_data()
    test_preprocess_data()
    test_train_pipeline()
    test_save_and_load_artifacts()
    test_single_and_batch_prediction()
    test_explain_and_business()
    print("\nAll tests passed.")
