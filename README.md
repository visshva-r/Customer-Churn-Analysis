# Telco Customer Churn Prediction

## Goal
Build a simple machine learning model to predict whether a customer will leave (Churn = Yes/No).

## Dataset
Used the Telco Customer Churn dataset from Kaggle which contains:
- Customer services used (internet, phone, etc.)
- Account details (tenure, contract, charges)
- Demographic information

## What I did
1. Cleaned the data  
   - Converted `TotalCharges` to numeric  
   - Removed missing values  
   - Dropped `customerID`

2. Preprocessing  
   - Encoded categorical columns using LabelEncoder  
   - Scaled features using StandardScaler  

3. Train/Validation Split  
   - 80% training, 20% validation  
   - Stratified split to keep class balance

## Models Tried
- Logistic Regression (baseline)
- Random Forest (improved model)

## Metrics Used
- Accuracy  
- Precision, Recall, F1-score  
- Confusion Matrix  

## Results (may vary slightly)
| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.82 |

Best performing model: **Logistic Regression**