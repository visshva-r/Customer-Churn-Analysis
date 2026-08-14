"""Model training, evaluation, and artifact persistence."""

import os
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.constants import RANDOM_STATE, TEST_SIZE, XGB_PARAMS
from src.data import load_raw_data, preprocess_data, project_root


def models_dir(base_dir: str | None = None) -> str:
    return os.path.join(base_dir or project_root(), "models")


def build_threshold_curve(y_true, y_proba, steps: int = 21) -> pd.DataFrame:
    thresholds = np.linspace(0.1, 0.9, steps)
    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        report = classification_report(
            y_true, preds, output_dict=True, zero_division=0
        )
        rows.append(
            {
                "threshold": t,
                "recall_churn": report["1"]["recall"],
                "precision_churn": report["1"]["precision"],
                "f1_churn": report["1"]["f1-score"],
                "flagged_pct": preds.mean(),
            }
        )
    return pd.DataFrame(rows)


def _evaluate_classifier(model, X_val_scaled, y_val) -> Dict[str, Any]:
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    return {
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "churn_recall": report["1"]["recall"],
        "churn_precision": report["1"]["precision"],
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "classification_report": report,
        "y_proba": y_proba,
    }


def train_pipeline(base_dir: str | None = None) -> Dict[str, Any]:
    """Train logistic baseline + tuned XGBoost and return all artifacts."""
    raw_df = load_raw_data()
    df, encoders = preprocess_data(raw_df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    feature_names = X.columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    logistic_model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train_res)

    xgb_model = XGBClassifier(**XGB_PARAMS)
    xgb_model.fit(X_train_scaled, y_train_res)

    logistic_eval = _evaluate_classifier(logistic_model, X_val_scaled, y_val)
    xgb_eval = _evaluate_classifier(xgb_model, X_val_scaled, y_val)

    metrics = {
        "accuracy": xgb_eval["accuracy"],
        "roc_auc": xgb_eval["roc_auc"],
        "churn_recall": xgb_eval["churn_recall"],
        "churn_precision": xgb_eval["churn_precision"],
        "confusion_matrix": xgb_eval["confusion_matrix"],
        "classification_report": xgb_eval["classification_report"],
        "feature_importance": pd.Series(
            xgb_model.feature_importances_, index=feature_names
        ).sort_values(ascending=False),
        "threshold_curve": build_threshold_curve(y_val, xgb_eval["y_proba"]),
        "baseline": {
            "accuracy": logistic_eval["accuracy"],
            "roc_auc": logistic_eval["roc_auc"],
            "churn_recall": logistic_eval["churn_recall"],
            "churn_precision": logistic_eval["churn_precision"],
        },
        "xgb": {
            "accuracy": xgb_eval["accuracy"],
            "roc_auc": xgb_eval["roc_auc"],
            "churn_recall": xgb_eval["churn_recall"],
            "churn_precision": xgb_eval["churn_precision"],
        },
    }

    return {
        "model": xgb_model,
        "logistic_model": logistic_model,
        "scaler": scaler,
        "encoders": encoders,
        "feature_names": feature_names,
        "metrics": metrics,
        "X_val_scaled": X_val_scaled,
    }


def save_artifacts(artifacts: Dict[str, Any], base_dir: str | None = None) -> str:
    out_dir = models_dir(base_dir)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(artifacts["model"], os.path.join(out_dir, "churn_model.joblib"))
    joblib.dump(
        artifacts["logistic_model"], os.path.join(out_dir, "logistic_model.joblib")
    )
    joblib.dump(artifacts["scaler"], os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(artifacts["encoders"], os.path.join(out_dir, "encoders.joblib"))
    joblib.dump(
        artifacts["feature_names"], os.path.join(out_dir, "feature_names.joblib")
    )
    joblib.dump(artifacts["metrics"], os.path.join(out_dir, "metrics.joblib"))
    return out_dir


def artifacts_exist(base_dir: str | None = None) -> bool:
    out_dir = models_dir(base_dir)
    required = [
        "churn_model.joblib",
        "scaler.joblib",
        "encoders.joblib",
        "feature_names.joblib",
        "metrics.joblib",
    ]
    return all(os.path.exists(os.path.join(out_dir, name)) for name in required)


def load_artifacts(base_dir: str | None = None) -> Tuple[Any, Any, Any, list, Dict]:
    out_dir = models_dir(base_dir)
    model = joblib.load(os.path.join(out_dir, "churn_model.joblib"))
    scaler = joblib.load(os.path.join(out_dir, "scaler.joblib"))
    encoders = joblib.load(os.path.join(out_dir, "encoders.joblib"))
    feature_names = joblib.load(os.path.join(out_dir, "feature_names.joblib"))
    metrics = joblib.load(os.path.join(out_dir, "metrics.joblib"))
    return model, scaler, encoders, feature_names, metrics


def load_or_train(base_dir: str | None = None) -> Tuple[Any, Any, Any, list, Dict]:
    if artifacts_exist(base_dir):
        return load_artifacts(base_dir)
    artifacts = train_pipeline(base_dir)
    save_artifacts(artifacts, base_dir)
    return (
        artifacts["model"],
        artifacts["scaler"],
        artifacts["encoders"],
        artifacts["feature_names"],
        artifacts["metrics"],
    )
