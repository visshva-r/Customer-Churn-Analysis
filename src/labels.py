"""Human-readable labels for features and business copy."""

FEATURE_DISPLAY = {
    "gender": "Gender",
    "SeniorCitizen": "Senior citizen",
    "Partner": "Partner status",
    "Dependents": "Dependents",
    "tenure": "Tenure",
    "PhoneService": "Phone service",
    "MultipleLines": "Multiple lines",
    "InternetService": "Internet service",
    "OnlineSecurity": "Online security",
    "OnlineBackup": "Online backup",
    "DeviceProtection": "Device protection",
    "TechSupport": "Tech support",
    "StreamingTV": "Streaming TV",
    "StreamingMovies": "Streaming movies",
    "Contract": "Contract",
    "PaperlessBilling": "Paperless billing",
    "PaymentMethod": "Payment method",
    "MonthlyCharges": "Monthly charges",
    "TotalCharges": "Total charges",
    "Tenure_Group": "Tenure segment",
    "Total_Services": "Add-on services count",
}


def format_feature(name: str, raw_input: dict | None = None) -> str:
    label = FEATURE_DISPLAY.get(name, name.replace("_", " ").title())
    if raw_input and name in raw_input:
        val = raw_input[name]
        if name == "SeniorCitizen":
            val = "Yes" if val == 1 else "No"
        return f"{label} ({val})"
    return label


def format_shap_series(shap_series, raw_input: dict) -> "pd.Series":
    import pandas as pd

    renamed = {format_feature(k, raw_input): v for k, v in shap_series.items()}
    return pd.Series(renamed)
