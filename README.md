# Churn Retention Scorer

Telco churn prediction with a recall-focused XGBoost model. Validation performance: ~79% churn recall, 0.81 ROC-AUC.

[![Live Demo](https://img.shields.io/badge/demo-streamlit.app-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://visshva-customer-churn-analysis.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

**Live app:** [visshva-customer-churn-analysis.streamlit.app](https://visshva-customer-churn-analysis.streamlit.app/)

## Overview

This project predicts which telco customers are likely to churn. The model is tuned for recall so more at-risk customers get flagged, even if that means contacting some who would have stayed.

The repo includes a reusable pipeline in `src/`, a Streamlit app, saved model artifacts, tests, and the original analysis notebook.

## App tabs

| Tab | What it does |
|-----|----------------|
| Score a customer | Run a single prediction with SHAP factors and next-step suggestions |
| Bulk scoring | Upload a CSV and download churn probabilities |
| Model performance | Compare Logistic Regression vs XGBoost on the validation set |
| Retention ROI | Rough estimate of outreach volume and revenue impact |

Use the example profiles in the sidebar to try a high-risk, stable, or new-customer scenario quickly.

## Pipeline

```
data/ -> src/data.py -> src/train.py -> models/ -> app.py
```

- SMOTE on the training set only
- Stratified 80/20 split
- Tuned XGBoost: `learning_rate=0.01`, `max_depth=3`, `n_estimators=100`
- Model artifacts saved with joblib for faster app startup

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

Telco Customer Churn dataset from Kaggle (~7,000 rows): service usage, billing, contract type, and demographics.
