"""Shared constants for the churn pipeline."""

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

TENURE_BINS = [0, 12, 24, 48, 60, float("inf")]
TENURE_LABELS = ["New", "Settled", "Loyal", "Very Loyal", "VIP"]

XGB_PARAMS = {
    "random_state": 42,
    "eval_metric": "logloss",
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.01,
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
DATA_FILENAME = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
