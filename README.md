# Churn Retention Scorer

**Catches ~79% of churners at 0.81 ROC-AUC** — live retention scoring demo for telco customer churn.

[![Live Demo](https://img.shields.io/badge/demo-streamlit.app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://visshva-customer-churn-analysis.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

**Live app:** [visshva-customer-churn-analysis.streamlit.app](https://visshva-customer-churn-analysis.streamlit.app/)

---

## What this project does

Predicts which telco customers are likely to churn, optimized for **recall** (catching at-risk users) rather than raw accuracy. Includes SHAP explanations, batch CSV scoring, and a retention ROI view for business stakeholders.

Built as a full pipeline (`src/`) with a deployed Streamlit front-end — same methodology as the analysis notebook.

## Demo highlights

| Tab | Purpose |
|-----|---------|
| **Score a customer** | Example profiles, risk gauge, SHAP drivers, retention plays |
| **Bulk scoring** | Upload CSV → export churn probabilities |
| **Model performance** | Logistic vs XGBoost, confusion matrix, threshold curves |
| **Retention ROI** | Outreach volume vs revenue protected (adjustable LTV) |

Try the **example profiles** in the sidebar (high-risk, stable loyal, new fiber) for instant demo scenarios.

## Architecture

```
data/ → src/data.py → src/train.py → models/ → app.py (Streamlit)
```

- SMOTE on training set only · stratified 80/20 split · tuned XGBoost (`lr=0.01`, `depth=3`, `n=100`)
- Saved joblib artifacts for fast cold starts on Streamlit Cloud

## Run locally

```bash
pip install -r requirements.txt
python scripts/train_and_save.py    # optional — models/ already included
python -m streamlit run app.py
python tests/test_app.py
```

## Results (validation set)

| Model | Accuracy | Churn recall | ROC-AUC |
|-------|----------|--------------|---------|
| Logistic Regression | 0.737 | 0.72 | 0.809 |
| **XGBoost + SMOTE** | **0.716** | **0.79** | **0.814** |

## Project layout

```
app.py                    Streamlit UI
src/                      Reusable ML pipeline
scripts/train_and_save.py Train + persist artifacts
models/                   Saved model, scaler, encoders
notebooks/                Full EDA + GridSearch notebook
data/                     Dataset + sample_customers.csv
tests/                    Pipeline tests + GitHub Actions CI
```

## Screenshots

Capture from the live app after deploy:
1. Score a customer (risk gauge + SHAP)
2. Retention ROI tab
3. Model performance comparison

Save to `docs/screenshots/` and embed here for portfolio polish.

## Deploy

Push to GitHub → [share.streamlit.io](https://share.streamlit.io) → main file `app.py`, Python 3.11.

## Dataset

Telco Customer Churn (Kaggle) — 7,000+ records with service usage, billing, contract, and demographics.
