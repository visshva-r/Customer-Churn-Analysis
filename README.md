# Telco Customer Churn Prediction Framework

**Live demo:** [https://visshva-customer-churn-analysis.streamlit.app/](https://visshva-customer-churn-analysis.streamlit.app/)

## Objective
Build an advanced machine learning pipeline to predict whether a customer will leave (Churn = Yes/No)[cite: 11]. Unlike standard baseline models that optimize purely for accuracy, this project focuses on maximizing **Recall** for the minority class to successfully identify high-risk churning customers.

## Interactive App
This repository includes a Streamlit UI (`app.py`) so non-technical users can explore the model.

**Try it online:** [https://visshva-customer-churn-analysis.streamlit.app/](https://visshva-customer-churn-analysis.streamlit.app/)

### How to run locally
1. **Install dependencies** (from project root):
   ```bash
   pip install -r requirements.txt
   ```
   Or install manually: `pip install streamlit xgboost scikit-learn imbalanced-learn pandas numpy`
2. From the project root, run (works even if `streamlit` is not on your PATH):
   ```bash
   python -m streamlit run app.py
   ```
3. A browser window will open where you can:
   - Enter a customer profile in the sidebar.
   - Use the **Predict for a customer** tab to get churn probability and prediction.
   - Use the **Model insights** tab to view validation metrics, feature importance, and threshold trade-offs.

### Testing
To verify the app logic (data load, preprocessing, model training, prediction) without starting the UI, run from the project root:
```bash
python tests/test_app.py
```
All checks should pass with no errors.

### Deploy on Streamlit Community Cloud
This app is deployed at [https://visshva-customer-churn-analysis.streamlit.app/](https://visshva-customer-churn-analysis.streamlit.app/). To redeploy or fork your own instance:

1. **Push this repo to GitHub** (if not already):  
   [https://github.com/visshva-r/Customer-Churn-Analysis](https://github.com/visshva-r/Customer-Churn-Analysis)

2. **Open Streamlit Community Cloud** and sign in with GitHub:  
   [https://share.streamlit.io](https://share.streamlit.io)

3. Click **Create app** → **From existing repo**.

4. Use these settings:
   | Setting | Value |
   |---------|--------|
   | Repository | `visshva-r/Customer-Churn-Analysis` |
   | Branch | `main` |
   | Main file path | `app.py` |
   | App URL (optional) | e.g. `customer-churn-analysis` |

5. Click **Deploy**. Streamlit installs packages from `requirements.txt` and starts `app.py`.

6. After deploy, your live URL will look like:  
   `https://<your-subdomain>.streamlit.app`  
   (This project: [https://visshva-customer-churn-analysis.streamlit.app/](https://visshva-customer-churn-analysis.streamlit.app/))

**Notes**
- The dataset in `data/` is included in the repo, so no extra secrets are needed.
- The model trains on first load (a few seconds); later visits use Streamlit cache.
- Python version on Cloud is usually 3.10–3.12; this project works with those versions.

## Project structure
- `app.py` — Streamlit demo (predict + model insights).
- `notebooks/churn_analysis.ipynb` — Full pipeline (EDA, feature engineering, baseline + XGBoost, ROC-AUC). Run from the `notebooks/` folder so `../data/` resolves correctly.
- `data/` — Telco Churn CSV (place dataset here).
- `requirements.txt` — Python dependencies for the app and notebook.
- `tests/test_app.py` — Script to test app logic without the UI.

## Dataset
Used the Telco Customer Churn dataset from Kaggle[cite: 11] (7,000+ records) containing:
- **Services used:** Internet, phone, multiple lines, tech support, etc.[cite: 11]
- **Account details:** Tenure, contract type, payment method, monthly/total charges.[cite: 11]
- **Demographic info:** Gender, senior citizen status, partners, and dependents.[cite: 11]

## Methodology & Pipeline
1. **Data Cleaning:** 
   - Converted `TotalCharges` to numeric and removed missing values[cite: 11]. 
   - Dropped non-predictive columns like `customerID`[cite: 11].
2. **Advanced Feature Engineering:**
   - **Lifecycle Binning:** Categorized the continuous `tenure` feature into distinct bins (`New`, `Settled`, `Loyal`, `Very Loyal`, `VIP`) to capture customer lifecycle stages.
   - **Service Aggregation:** Engineered a `Total_Services` feature to quantify user engagement across the platform.
3. **Handling Data Imbalance:**
   - Identified a severe 73:27 class imbalance.
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** exclusively on the training set to prevent data leakage and model bias toward the majority class.
4. **Preprocessing:** 
   - Encoded categorical features using `LabelEncoder`[cite: 11].
   - Scaled continuous features using `StandardScaler`[cite: 11].
   - 80/20 Stratified Train/Validation split[cite: 11].

## Models & Tuning
- **Baseline Model:** Logistic Regression[cite: 11].
- **Advanced Model:** XGBoost Classifier.
- **Hyperparameter Tuning:** Utilized `GridSearchCV` specifically optimizing for **Recall**, iterating over `n_estimators`, `max_depth`, and `learning_rate` to find the optimal architecture.

## Evaluation Metrics
- **ROC-AUC Score** (Primary performance indicator)
- Recall, Precision, F1-Score[cite: 11]
- Confusion Matrix[cite: 11]

## Key Results & Business Impact
| Model | Accuracy | Churner Recall (Class 1) | ROC-AUC |
|-------|----------|--------------------------|---------|
| Logistic Regression (Baseline) | ~73.7% | 0.72 | - |
| **XGBoost (Tuned + SMOTE)** | **~71.5%** | **0.79** | **0.8136** |

**Conclusion:**
While raw accuracy[cite: 11] saw a slight dip, the tuned XGBoost model combined with SMOTE successfully boosted the **Recall for churners to 79%**. In a real-world business context, this trade-off is highly desirable: the model correctly flags nearly 80% of at-risk customers, allowing targeted retention strategies, backed by a strong ROC-AUC score of 0.814 indicating robust predictive power.
