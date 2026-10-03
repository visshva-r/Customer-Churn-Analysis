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

Holdout set: stratified 20% split (`random_state=42`). Same numbers shown in the Streamlit app (loaded from `models/metrics.joblib`).

| Model | Accuracy | Churn recall | ROC-AUC |
|-------|----------|--------------|---------|
| Logistic Regression | 0.737 | 0.717 | 0.809 |
| XGBoost + SMOTE | 0.716 | 0.789 | 0.814 |

Rounded for resume talking points: **~79% churn recall**, **0.81 ROC-AUC** on the tuned XGBoost model.

## How I optimized recall (interviews)

1. **Problem framing:** Churn is imbalanced (~73% stay / ~27% churn). Missing a churner is often costlier than a false alarm, so the goal was high **recall on the churn class**, not maximum accuracy.

2. **Resampling without leakage:** Applied **SMOTE only on the training set** after an 80/20 stratified split so the validation set stays realistic.

3. **Hyperparameter search:** In `notebooks/churn_analysis.ipynb`, used **GridSearchCV with `scoring='recall'`** over `n_estimators`, `max_depth`, and `learning_rate`. Best params (`lr=0.01`, `depth=3`, `n=100`) are fixed in `src/constants.py` for the app and `scripts/train_and_save.py`.

4. **Model choice:** Compared a **Logistic Regression** baseline to **XGBoost**. XGBoost trades a small accuracy drop for higher churn recall (0.717 → 0.789) while keeping strong ranking performance (ROC-AUC 0.814).

5. **Operating point:** Default classification uses threshold 0.5; the app includes a **threshold slider** and ROI tab to discuss precision vs recall vs how many customers you contact.

**Live demo:** [visshva-customer-churn-analysis.streamlit.app](https://visshva-customer-churn-analysis.streamlit.app/) — scores single customers, batch CSV, SHAP factors, and model comparison.

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
