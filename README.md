# Telco Customer Churn Prediction Framework

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-red?style=for-the-badge&logo=streamlit)](https://visshva-customer-churn-analysis.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/visshva-r/Customer-Churn-Analysis)

**Live demo:** [https://visshva-customer-churn-analysis.streamlit.app/](https://visshva-customer-churn-analysis.streamlit.app/)

## Objective
Build an advanced machine learning pipeline to predict whether a customer will leave (Churn = Yes/No). Unlike baseline models that optimize purely for accuracy, this project focuses on maximizing **recall** for the minority class to identify high-risk churning customers.

## Key features
- **Tuned XGBoost + SMOTE** — recall-focused churn detection (~79% churn recall, ~0.81 ROC-AUC)
- **SHAP explainability** — top factors driving each customer's churn risk
- **Business impact tab** — threshold trade-offs in plain business language
- **Batch CSV prediction** — upload customers, download scores
- **Baseline vs XGBoost comparison** — Logistic Regression vs tuned model in the app
- **Reusable `src/` pipeline** — shared logic for app, training script, and tests
- **Model persistence** — saved artifacts in `models/` for fast startup
- **CI tests** — GitHub Actions runs `tests/test_app.py` on push
- **Live Streamlit deployment** — public demo link above

## Architecture

```
data/ (CSV)
   ↓
src/data.py        → clean + feature engineering + encoding
   ↓
src/train.py       → SMOTE → StandardScaler → Logistic + XGBoost
   ↓
models/            → joblib artifacts (model, scaler, encoders, metrics)
   ↓
app.py (Streamlit) → Predict | Batch | Model insights | Business impact
```

The notebook (`notebooks/churn_analysis.ipynb`) follows the same methodology; the app uses the tuned hyperparameters from GridSearch (`learning_rate=0.01`, `max_depth=3`, `n_estimators=100`).

## Interactive app

**Try it online:** [https://visshva-customer-churn-analysis.streamlit.app/](https://visshva-customer-churn-analysis.streamlit.app/)

| Tab | Description |
|-----|-------------|
| **Predict** | Single-customer churn score + SHAP explanation + retention tips |
| **Batch upload** | CSV in → predictions out |
| **Model insights** | Baseline vs XGBoost, confusion matrix, feature importance |
| **Business impact** | Recall vs outreach cost at your chosen threshold |

### Run locally
```bash
pip install -r requirements.txt
python scripts/train_and_save.py   # optional: pre-train and save models/
python -m streamlit run app.py
```

### Train and save model artifacts
```bash
python scripts/train_and_save.py
```
Artifacts are written to `models/` and loaded automatically by the app (falls back to training if missing).

### Run tests
```bash
python tests/test_app.py
```

### Deploy on Streamlit Community Cloud
1. Push repo to GitHub: [visshva-r/Customer-Churn-Analysis](https://github.com/visshva-r/Customer-Churn-Analysis)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **Create app** → select repo, branch `main`, main file `app.py`
3. Use Python **3.11** (recommended). No secrets required.
4. Redeploy after pushing updates to `main`.

## Project structure
```
├── app.py                      # Streamlit demo
├── src/
│   ├── data.py                 # Load, clean, feature engineering
│   ├── train.py                # Train, evaluate, save/load artifacts
│   ├── predict.py              # Single + batch prediction
│   ├── explain.py              # SHAP + plain-English summaries
│   └── business.py             # Business impact metrics
├── scripts/train_and_save.py   # CLI to train and persist models
├── models/                     # Saved joblib artifacts
├── notebooks/churn_analysis.ipynb
├── data/WA_Fn-UseC_-Telco-Customer-Churn.csv
├── tests/test_app.py
├── requirements.txt
└── .github/workflows/test.yml
```

## Screenshots
Add screenshots of the live app tabs here after deployment (Predict, Model insights, Business impact).

## Dataset
Telco Customer Churn dataset from Kaggle (7,000+ records):
- **Services:** Internet, phone, tech support, streaming, etc.
- **Account:** Tenure, contract, payment method, charges
- **Demographics:** Gender, senior citizen, partner, dependents

## Methodology
1. **Data cleaning** — numeric `TotalCharges`, drop missing rows, remove `customerID`
2. **Feature engineering** — tenure bins (`New` → `VIP`), `Total_Services` count
3. **Imbalance** — SMOTE on training set only (73:27 split)
4. **Preprocessing** — LabelEncoder + StandardScaler, 80/20 stratified split
5. **Models** — Logistic Regression baseline + tuned XGBoost (recall-optimized)

## Key results
| Model | Accuracy | Churn recall | ROC-AUC |
|-------|----------|--------------|---------|
| Logistic Regression | ~73.7% | 0.72 | — |
| **XGBoost (tuned + SMOTE)** | **~71.5%** | **0.79** | **0.8136** |

The tuned XGBoost model trades a small accuracy drop for catching ~79% of churners — a desirable trade-off for retention campaigns.
