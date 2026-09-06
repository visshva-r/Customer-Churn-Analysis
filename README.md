# Customer Churn Analysis

Predicts which customers are likely to churn using a recall-focused XGBoost model. Validation: ~79% churn recall, 0.81 ROC-AUC.

[![Live Demo](https://img.shields.io/badge/demo-streamlit.app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://visshva-customer-churn-analysis.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

**Live app:** [visshva-customer-churn-analysis.streamlit.app](https://visshva-customer-churn-analysis.streamlit.app/)

## Overview

This project builds a machine learning pipeline to identify customers at risk of leaving. The model is tuned for recall so more churners are caught, even if that means flagging some customers who would have stayed.

Includes a reusable pipeline in `src/`, a Streamlit app, saved model artifacts, tests, and an analysis notebook.

## App features

| Tab | Description |
|-----|-------------|
| Score a customer | Single prediction with SHAP feature contributions and next-step suggestions |
| Bulk scoring | Upload a CSV and download churn probabilities |
| Model performance | Logistic Regression vs XGBoost on the validation set |
| Retention ROI | Estimate outreach volume and revenue impact at a chosen threshold |

Sample customer profiles are available in the sidebar for testing.

## Pipeline

```
data/ -> src/data.py -> src/train.py -> models/ -> app.py
```

- SMOTE applied on the training set only
- Stratified 80/20 train/validation split
- Tuned XGBoost: `learning_rate=0.01`, `max_depth=3`, `n_estimators=100`
- Model artifacts saved with joblib

## Run locally

```bash
pip install -r requirements.txt
python scripts/train_and_save.py    # optional; models/ is already in the repo
python -m streamlit run app.py
python tests/test_app.py
```

## Validation results

| Model | Accuracy | Churn recall | ROC-AUC |
|-------|----------|--------------|---------|
| Logistic Regression | 0.737 | 0.72 | 0.809 |
| XGBoost + SMOTE | 0.716 | 0.79 | 0.814 |

## Project layout

```
app.py                    Streamlit UI
src/                      ML pipeline modules
scripts/train_and_save.py Train and save artifacts
models/                   Saved model, scaler, encoders
notebooks/                EDA and GridSearch notebook
data/                     Dataset and sample_customers.csv
tests/                    Pipeline tests (GitHub Actions on push)
```

## Deploy

Push to GitHub, then deploy on [share.streamlit.io](https://share.streamlit.io) with main file `app.py` and Python 3.11.

## Dataset

IBM Telco Customer Churn dataset (~7,000 rows): service usage, billing, contract type, and demographics.
