import os

import numpy as np
import pandas as pd
import streamlit as st
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df: pd.DataFrame):
    df = df.copy()

    # Convert TotalCharges and drop missing
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    # Drop identifier
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Tenure binning
    df["Tenure_Group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=["New", "Settled", "Loyal", "Very Loyal", "VIP"],
    )

    # Total services feature
    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_columns:
        flag_col = f"{col}_Flag"
        df[flag_col] = df[col].apply(lambda x: 1 if x == "Yes" else 0)
    df["Total_Services"] = df[[f"{c}_Flag" for c in service_columns]].sum(axis=1)
    df.drop(columns=[f"{c}_Flag" for c in service_columns], inplace=True)

    # Encode categoricals (including Tenure_Group, Churn)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    return df, encoders


@st.cache_resource
def train_model():
    raw_df = load_raw_data()
    df, encoders = preprocess_data(raw_df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_val_scaled = scaler.transform(X_val)

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_estimators=100,
        max_depth=3,
        learning_rate=0.01,
    )
    model.fit(X_train_scaled, y_train_res)

    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "roc_auc": roc_auc_score(y_val, y_proba),
        "confusion_matrix": confusion_matrix(y_val, y_pred),
        "classification_report": classification_report(y_val, y_pred, output_dict=True),
        "feature_importance": pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False),
        "threshold_curve": _build_threshold_curve(y_val, y_proba),
    }

    return model, scaler, encoders, X.columns.tolist(), metrics


def _build_threshold_curve(y_true, y_proba, steps=21):
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


def build_user_input(encoders, feature_names):
    st.sidebar.header("Customer Profile")

    raw_df = load_raw_data()

    input_data = {}

    def cat_select(label, column):
        options = sorted(raw_df[column].unique().tolist())
        default = options[0]
        return st.sidebar.selectbox(label, options, index=options.index(default))

    def num_input(label, column, step=1.0):
        # Ensure numeric stats even if the original column has spaces or strings
        col_data = pd.to_numeric(raw_df[column], errors="coerce").dropna()
        return st.sidebar.number_input(
            label,
            float(col_data.min()),
            float(col_data.max()),
            float(col_data.median()),
            step=step,
        )

    input_data["gender"] = cat_select("Gender", "gender")
    senior_choice = st.sidebar.selectbox(
        "Senior Citizen", ["No", "Yes"], index=0
    )
    senior_map = {"No": 0, "Yes": 1}
    input_data["SeniorCitizen"] = senior_map[senior_choice]
    input_data["Partner"] = cat_select("Partner", "Partner")
    input_data["Dependents"] = cat_select("Dependents", "Dependents")
    input_data["tenure"] = num_input("Tenure (months)", "tenure", step=1.0)
    input_data["PhoneService"] = cat_select("Phone Service", "PhoneService")
    input_data["MultipleLines"] = cat_select("Multiple Lines", "MultipleLines")
    input_data["InternetService"] = cat_select("Internet Service", "InternetService")
    input_data["OnlineSecurity"] = cat_select("Online Security", "OnlineSecurity")
    input_data["OnlineBackup"] = cat_select("Online Backup", "OnlineBackup")
    input_data["DeviceProtection"] = cat_select(
        "Device Protection", "DeviceProtection"
    )
    input_data["TechSupport"] = cat_select("Tech Support", "TechSupport")
    input_data["StreamingTV"] = cat_select("Streaming TV", "StreamingTV")
    input_data["StreamingMovies"] = cat_select(
        "Streaming Movies", "StreamingMovies"
    )
    input_data["Contract"] = cat_select("Contract", "Contract")
    input_data["PaperlessBilling"] = cat_select(
        "Paperless Billing", "PaperlessBilling"
    )
    input_data["PaymentMethod"] = cat_select("Payment Method", "PaymentMethod")
    input_data["MonthlyCharges"] = num_input(
        "Monthly Charges", "MonthlyCharges", step=1.0
    )
    input_data["TotalCharges"] = num_input(
        "Total Charges", "TotalCharges", step=10.0
    )

    row = pd.DataFrame([input_data])

    # Recreate engineered features
    row["Tenure_Group"] = pd.cut(
        row["tenure"],
        bins=[0, 12, 24, 48, 60, np.inf],
        labels=["New", "Settled", "Loyal", "Very Loyal", "VIP"],
    )

    service_columns = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    for col in service_columns:
        row[f"{col}_Flag"] = row[col].apply(lambda x: 1 if x == "Yes" else 0)
    row["Total_Services"] = row[[f"{c}_Flag" for c in service_columns]].sum(axis=1)
    row.drop(columns=[f"{c}_Flag" for c in service_columns], inplace=True)

    # Apply encoders
    for col, le in encoders.items():
        if col in row.columns:
            # Fall back safely if unseen category
            def safe_transform(val):
                if val in le.classes_:
                    return le.transform([val])[0]
                return le.transform([le.classes_[0]])[0]

            row[col] = row[col].apply(safe_transform)

    # Ensure all features exist
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    row = row[feature_names]
    return row


def main():
    st.set_page_config(
        page_title="Telco Customer Churn Predictor", layout="wide"
    )
    st.title("Telco Customer Churn Prediction")
    st.markdown(
        "This app uses an XGBoost model optimized for **recall** on churners "
        "to help identify high-risk customers."
    )

    with st.spinner("Training / loading model..."):
        model, scaler, encoders, feature_names, metrics = train_model()

    # Sidebar controls common to the page
    st.sidebar.markdown("### Prediction Settings")
    decision_threshold = st.sidebar.slider(
        "Churn decision threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Customers with predicted probability above this threshold are labeled as churners.",
    )

    tab_predict, tab_metrics = st.tabs(["🔮 Predict for a customer", "📊 Model insights"])

    with tab_metrics:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Validation Performance")
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

            cm = metrics["confusion_matrix"]
            st.markdown("**Confusion Matrix** (rows = actual, cols = predicted)")
            st.dataframe(pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"]))

        with col2:
            st.subheader("Feature Importance (XGBoost)")
            fi = metrics["feature_importance"].head(10).sort_values()
            st.bar_chart(fi)

        st.subheader("Threshold Trade-offs (Class 1 = churn)")
        th_df = metrics["threshold_curve"]
        st.line_chart(
            th_df.set_index("threshold")[["recall_churn", "precision_churn", "flagged_pct"]]
        )

    with tab_predict:
        user_features = build_user_input(encoders, feature_names)

        if st.sidebar.button("Predict Churn Risk"):
            scaled = scaler.transform(user_features)
            proba = model.predict_proba(scaled)[0, 1]
            pred = "Yes" if proba >= decision_threshold else "No"

            st.subheader("Prediction")
            st.write(f"**Will the customer churn?** {pred}")
            st.write(f"**Churn probability:** {proba:.2%}")
            st.write(f"**Decision threshold:** {decision_threshold:.2f}")

            if pred == "Yes":
                st.info(
                    "This customer is predicted to **churn** at the selected threshold. "
                    "Consider targeted retention actions (discounts, outreach, contract changes)."
                )
            else:
                st.success(
                    "This customer is predicted to **stay** under current conditions "
                    f"at a threshold of {decision_threshold:.2f}."
                )


if __name__ == "__main__":
    main()

