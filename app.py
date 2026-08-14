import pandas as pd
import streamlit as st

from src.business import business_message, threshold_summary
from src.data import load_raw_data
from src.explain import explain_prediction, retention_recommendations
from src.predict import predict_batch, predict_single
from src.train import load_or_train


@st.cache_resource
def get_artifacts():
    return load_or_train()


def build_user_input(raw_df: pd.DataFrame):
    st.sidebar.header("Customer Profile")
    input_data = {}

    def cat_select(label, column):
        options = sorted(raw_df[column].unique().tolist())
        return st.sidebar.selectbox(label, options, index=0)

    def num_input(label, column, step=1.0):
        col_data = pd.to_numeric(raw_df[column], errors="coerce").dropna()
        return st.sidebar.number_input(
            label,
            float(col_data.min()),
            float(col_data.max()),
            float(col_data.median()),
            step=step,
        )

    input_data["gender"] = cat_select("Gender", "gender")
    senior_choice = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"], index=0)
    input_data["SeniorCitizen"] = {"No": 0, "Yes": 1}[senior_choice]
    input_data["Partner"] = cat_select("Partner", "Partner")
    input_data["Dependents"] = cat_select("Dependents", "Dependents")
    input_data["tenure"] = num_input("Tenure (months)", "tenure", step=1.0)
    input_data["PhoneService"] = cat_select("Phone Service", "PhoneService")
    input_data["MultipleLines"] = cat_select("Multiple Lines", "MultipleLines")
    input_data["InternetService"] = cat_select("Internet Service", "InternetService")
    input_data["OnlineSecurity"] = cat_select("Online Security", "OnlineSecurity")
    input_data["OnlineBackup"] = cat_select("Online Backup", "OnlineBackup")
    input_data["DeviceProtection"] = cat_select("Device Protection", "DeviceProtection")
    input_data["TechSupport"] = cat_select("Tech Support", "TechSupport")
    input_data["StreamingTV"] = cat_select("Streaming TV", "StreamingTV")
    input_data["StreamingMovies"] = cat_select("Streaming Movies", "StreamingMovies")
    input_data["Contract"] = cat_select("Contract", "Contract")
    input_data["PaperlessBilling"] = cat_select("Paperless Billing", "PaperlessBilling")
    input_data["PaymentMethod"] = cat_select("Payment Method", "PaymentMethod")
    input_data["MonthlyCharges"] = num_input("Monthly Charges", "MonthlyCharges", step=1.0)
    input_data["TotalCharges"] = num_input("Total Charges", "TotalCharges", step=10.0)
    return input_data


def main():
    st.set_page_config(page_title="Telco Customer Churn Predictor", layout="wide")
    st.title("Telco Customer Churn Prediction")
    st.markdown(
        "XGBoost model optimized for **recall** on churners — identify at-risk "
        "customers and act before they leave."
    )

    with st.spinner("Loading model..."):
        model, scaler, encoders, feature_names, metrics = get_artifacts()

    st.sidebar.markdown("### Prediction Settings")
    decision_threshold = st.sidebar.slider(
        "Churn decision threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Customers with probability above this threshold are labeled as churners.",
    )

    tab_predict, tab_batch, tab_metrics, tab_business = st.tabs(
        [
            "Predict",
            "Batch upload",
            "Model insights",
            "Business impact",
        ]
    )

    raw_df = load_raw_data()

    with tab_metrics:
        st.subheader("Baseline vs XGBoost (validation set)")
        comparison = pd.DataFrame(
            {
                "Logistic Regression": metrics["baseline"],
                "XGBoost (tuned)": metrics["xgb"],
            }
        ).T[["accuracy", "churn_recall", "churn_precision", "roc_auc"]]
        comparison.columns = ["Accuracy", "Churn recall", "Churn precision", "ROC-AUC"]
        st.dataframe(comparison.round(3), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("XGBoost validation metrics")
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            st.metric("Churn recall", f"{metrics['churn_recall']:.3f}")
            cm = metrics["confusion_matrix"]
            st.markdown("**Confusion matrix** (rows = actual, cols = predicted)")
            st.dataframe(
                pd.DataFrame(cm, index=["No", "Yes"], columns=["No", "Yes"])
            )
        with col2:
            st.subheader("Feature importance (top 10)")
            fi = metrics["feature_importance"].head(10).sort_values()
            st.bar_chart(fi)

        st.subheader("Threshold trade-offs")
        th_df = metrics["threshold_curve"]
        st.line_chart(
            th_df.set_index("threshold")[
                ["recall_churn", "precision_churn", "flagged_pct"]
            ]
        )

    with tab_business:
        st.subheader("Business impact at selected threshold")
        summary = threshold_summary(metrics["threshold_curve"], decision_threshold)
        st.markdown(business_message(decision_threshold, summary))
        c1, c2, c3 = st.columns(3)
        c1.metric("Churners caught", f"{summary['recall_churn_pct']:.1f}%")
        c2.metric("Customers flagged", f"{summary['flagged_pct']:.1f}%")
        c3.metric("Churn precision", f"{summary['precision_churn_pct']:.1f}%")
        st.markdown(
            """
**Why recall matters for churn:** Missing a churner (false negative) often costs more
than contacting a loyal customer (false positive). This model prioritizes catching
at-risk customers so retention teams can intervene early.
            """
        )

    with tab_predict:
        user_input = build_user_input(raw_df)
        if st.sidebar.button("Predict churn risk", type="primary"):
            proba, pred, features = predict_single(
                model,
                scaler,
                encoders,
                feature_names,
                user_input,
                decision_threshold,
            )
            st.subheader("Prediction")
            st.write(f"**Will the customer churn?** {pred}")
            st.write(f"**Churn probability:** {proba:.2%}")
            st.write(f"**Decision threshold:** {decision_threshold:.2f}")

            top_factors, summary = explain_prediction(
                model, scaler, features, feature_names, user_input
            )
            st.markdown(f"**Explanation:** {summary}")
            if not top_factors.empty:
                st.markdown("**Top contributing features (SHAP):**")
                st.bar_chart(top_factors.sort_values())

            recs = retention_recommendations(user_input, pred)
            st.markdown("**Recommended actions:**")
            for rec in recs:
                st.markdown(f"- {rec}")

            if pred == "Yes":
                st.info("Flag for retention outreach within 7 days.")
            else:
                st.success("No immediate retention action required.")

    with tab_batch:
        st.markdown(
            "Upload a CSV with the same columns as the training data "
            "(Churn column optional). Invalid rows are dropped during cleaning."
        )
        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(batch_df)}** rows.")
            results = predict_batch(
                model,
                scaler,
                encoders,
                feature_names,
                batch_df,
                decision_threshold,
            )
            st.dataframe(results.head(20), use_container_width=True)
            st.download_button(
                "Download predictions CSV",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
