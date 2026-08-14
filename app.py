import os

import pandas as pd
import streamlit as st

from src.business import (
    business_message,
    retention_roi_estimate,
    threshold_summary,
)
from src.data import load_raw_data, project_root
from src.explain import explain_prediction, retention_recommendations
from src.predict import predict_batch, predict_single
from src.presets import PRESETS
from src.train import load_or_train

CUSTOM_CSS = """
<style>
    .hero {
        background: linear-gradient(135deg, #0B3D91 0%, #1a5fbf 100%);
        color: white;
        padding: 1.25rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1.25rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 1.75rem; }
    .hero p { margin: 0.35rem 0 0 0; opacity: 0.92; }
    .risk-card {
        background: #ffffff;
        border: 1px solid #dde4ee;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin: 0.75rem 0;
    }
    .risk-low { border-left: 5px solid #2ecc71; }
    .risk-mid { border-left: 5px solid #f39c12; }
    .risk-high { border-left: 5px solid #e74c3c; }
    div[data-testid="stSidebar"] { background-color: #f4f7fb; }
</style>
"""


@st.cache_resource
def get_artifacts():
    return load_or_train()


def risk_band(proba: float) -> tuple[str, str]:
    if proba >= 0.65:
        return "High", "risk-high"
    if proba >= 0.4:
        return "Medium", "risk-mid"
    return "Low", "risk-low"


def render_risk_gauge(proba: float, label: str):
    band, css_class = risk_band(proba)
    st.markdown(
        f"""
        <div class="risk-card {css_class}">
            <strong>Churn risk: {band}</strong> &nbsp;·&nbsp; {proba:.1%} probability
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(max(proba, 0.0), 1.0))
    st.caption(f"Prediction at current threshold: **{label}**")


def build_user_input(
    raw_df: pd.DataFrame, defaults: dict | None = None, preset_key: str = "Custom"
) -> dict:
    defaults = defaults or {}
    input_data = {}

    def cat_select(label, column, container):
        options = sorted(raw_df[column].unique().tolist())
        default = defaults.get(column, options[0])
        idx = options.index(default) if default in options else 0
        return container.selectbox(
            label, options, index=idx, key=f"{preset_key}_{column}"
        )

    def num_input(label, column, container, step=1.0):
        col_data = pd.to_numeric(raw_df[column], errors="coerce").dropna()
        default = float(defaults.get(column, col_data.median()))
        return container.number_input(
            label,
            float(col_data.min()),
            float(col_data.max()),
            default,
            step=step,
            key=f"{preset_key}_{column}",
        )

    with st.sidebar.expander("Demographics", expanded=True):
        input_data["gender"] = cat_select("Gender", "gender", st)
        senior_default = "Yes" if defaults.get("SeniorCitizen") == 1 else "No"
        senior_options = ["No", "Yes"]
        senior_idx = senior_options.index(senior_default) if senior_default in senior_options else 0
        senior_choice = st.selectbox(
            "Senior citizen",
            senior_options,
            index=senior_idx,
            key=f"{preset_key}_SeniorCitizen",
        )
        input_data["SeniorCitizen"] = {"No": 0, "Yes": 1}[senior_choice]
        input_data["Partner"] = cat_select("Partner", "Partner", st)
        input_data["Dependents"] = cat_select("Dependents", "Dependents", st)
        input_data["tenure"] = num_input("Tenure (months)", "tenure", st, step=1.0)

    with st.sidebar.expander("Services", expanded=False):
        input_data["PhoneService"] = cat_select("Phone service", "PhoneService", st)
        input_data["MultipleLines"] = cat_select("Multiple lines", "MultipleLines", st)
        input_data["InternetService"] = cat_select("Internet service", "InternetService", st)
        input_data["OnlineSecurity"] = cat_select("Online security", "OnlineSecurity", st)
        input_data["OnlineBackup"] = cat_select("Online backup", "OnlineBackup", st)
        input_data["DeviceProtection"] = cat_select("Device protection", "DeviceProtection", st)
        input_data["TechSupport"] = cat_select("Tech support", "TechSupport", st)
        input_data["StreamingTV"] = cat_select("Streaming TV", "StreamingTV", st)
        input_data["StreamingMovies"] = cat_select("Streaming movies", "StreamingMovies", st)

    with st.sidebar.expander("Billing & contract", expanded=False):
        input_data["Contract"] = cat_select("Contract", "Contract", st)
        input_data["PaperlessBilling"] = cat_select("Paperless billing", "PaperlessBilling", st)
        input_data["PaymentMethod"] = cat_select("Payment method", "PaymentMethod", st)
        input_data["MonthlyCharges"] = num_input("Monthly charges ($)", "MonthlyCharges", st, step=1.0)
        input_data["TotalCharges"] = num_input("Total charges ($)", "TotalCharges", st, step=10.0)

    return input_data


def main():
    st.set_page_config(page_title="Churn Retention Scorer", layout="wide", page_icon="📡")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
            <h1>Churn Retention Scorer</h1>
            <p>Catches ~79% of churners · ROC-AUC 0.81 · Built for retention teams, not just accuracy</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading model..."):
        model, scaler, encoders, feature_names, metrics = get_artifacts()

    st.sidebar.markdown("### Scoring settings")
    decision_threshold = st.sidebar.slider(
        "Flag customers above",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        format="%.2f",
    )

    preset_choice = st.sidebar.selectbox(
        "Example profile",
        ["Custom"] + list(PRESETS.keys()),
        help="Load a realistic customer scenario for demos.",
    )
    profile_defaults = PRESETS.get(preset_choice)

    tab_score, tab_bulk, tab_model, tab_roi = st.tabs(
        ["Score a customer", "Bulk scoring", "Model performance", "Retention ROI"]
    )

    raw_df = load_raw_data()

    with tab_model:
        st.markdown("Validation set comparison — same split and tuned hyperparameters as the notebook.")
        comparison = pd.DataFrame(
            {"Logistic Regression": metrics["baseline"], "XGBoost (tuned)": metrics["xgb"]}
        ).T[["accuracy", "churn_recall", "churn_precision", "roc_auc"]]
        comparison.columns = ["Accuracy", "Churn recall", "Churn precision", "ROC-AUC"]
        st.dataframe(comparison.round(3), use_container_width=True, hide_index=False)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            st.metric("Churn recall", f"{metrics['churn_recall']:.3f}")
            cm = metrics["confusion_matrix"]
            st.caption("Confusion matrix (actual → predicted)")
            st.dataframe(pd.DataFrame(cm, index=["Stay", "Churn"], columns=["Stay", "Churn"]))
        with col2:
            fi = metrics["feature_importance"].head(8).sort_values()
            st.caption("What the model weighs most")
            st.bar_chart(fi)

        th_df = metrics["threshold_curve"]
        st.caption("Threshold trade-off curve")
        st.line_chart(
            th_df.set_index("threshold")[["recall_churn", "precision_churn", "flagged_pct"]]
        )

    with tab_roi:
        st.markdown("Translate model output into outreach volume and revenue at risk.")
        summary = threshold_summary(metrics["threshold_curve"], decision_threshold)
        st.markdown(business_message(decision_threshold, summary))

        portfolio_size = st.number_input("Portfolio size (customers)", 100, 50000, 1000, step=100)
        ltv = st.number_input("Assumed LTV per churner ($)", 50, 5000, 500, step=50)
        roi = retention_roi_estimate(summary, portfolio_size, ltv)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Churners caught", f"{roi['churners_caught']:.0f}")
        c2.metric("Customers flagged", f"{roi['customers_flagged']:.0f}")
        c3.metric("Revenue protected", f"${roi['revenue_protected']:,.0f}")
        c4.metric("Revenue still at risk", f"${roi['missed_revenue']:,.0f}")

        st.caption(
            f"Illustrative model: ~{portfolio_size} customers, "
            f"~27% historical churn rate, ${ltv:,.0f} LTV per lost customer. "
            "Adjust inputs to match your business."
        )

    with tab_score:
        st.sidebar.markdown("### Customer profile")
        user_input = build_user_input(raw_df, profile_defaults, preset_choice)

        if st.sidebar.button("Run churn score", type="primary", use_container_width=True):
            proba, pred, features = predict_single(
                model, scaler, encoders, feature_names, user_input, decision_threshold
            )

            col_a, col_b = st.columns([1, 2])
            with col_a:
                render_risk_gauge(proba, pred)
            with col_b:
                top_factors, summary = explain_prediction(
                    model, scaler, features, feature_names, user_input
                )
                st.markdown(f"**Why this score?** {summary}")
                if not top_factors.empty:
                    st.caption("SHAP contribution (top factors)")
                    st.bar_chart(top_factors.sort_values())

            st.markdown("**Suggested retention plays**")
            recs = retention_recommendations(user_input, pred, proba)
            for rec in recs:
                st.markdown(f"- {rec}")

    with tab_bulk:
        st.markdown("Upload a customer list (same columns as training data; `Churn` optional).")
        sample_path = os.path.join(project_root(), "data", "sample_customers.csv")
        with open(sample_path, "rb") as f:
            st.download_button(
                "Download sample CSV (5 rows)",
                data=f,
                file_name="sample_customers.csv",
                mime="text/csv",
            )

        uploaded = st.file_uploader("Customer CSV", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded)
            st.caption(f"{len(batch_df)} rows loaded")
            results = predict_batch(
                model, scaler, encoders, feature_names, batch_df, decision_threshold
            )
            high_risk = (results["Churn_Probability"] >= decision_threshold).sum()
            st.metric("Flagged for outreach", f"{high_risk} / {len(results)}")
            st.dataframe(results.head(25), use_container_width=True)
            st.download_button(
                "Export scores",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="churn_scores.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
