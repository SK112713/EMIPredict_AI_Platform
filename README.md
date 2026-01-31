# EMI Prediction System

## Project Overview
The **EMI Prediction System** is a data-driven platform designed to assist banks and financial institutions in assessing customer eligibility for EMI-based loans and predicting the maximum EMI amount a customer can safely afford.  
This system uses **Machine Learning (classification and regression)** along with **MLflow for model tracking and deployment** to make accurate predictions based on financial and demographic data.

---

## Motivation
Many customers face challenges in loan approvals due to incomplete, inconsistent, or imbalanced financial data.  
This project aims to automate the EMI eligibility assessment process, reduce risk for lenders, and help customers understand their borrowing potential.

---

## Dataset
The dataset contains information about customers, including:

**Numeric Features**:
- `age`, `monthly_salary`, `bank_balance`, `current_emi_amount`, `requested_amount`, `requested_tenure`, `credit_score`, etc.

**Categorical Features**:
- `gender`, `marital_status`, `education`, `employment_type`, `company_type`, `house_type`, `existing_loans`, `emi_scenario`

**Target Variables**:
- **Classification**: `emi_eligibility` (e.g., Low_Risk, High_Risk)  
- **Regression**: `max_monthly_emi` (maximum EMI the customer can afford)

---

## Data Preprocessing
The dataset underwent extensive cleaning and preprocessing:

1. **Handling missing values**:
   - Numeric columns: filled using **median per group** or overall mean.
   - Categorical columns: filled using **mode**.

2. **Data type conversions**:
   - Ensured numeric columns (`age`, `monthly_salary`, `bank_balance`, etc.) are of type `float` or `int`.
   - Categorical columns converted to type `category`.

3. **Outlier handling**:
   - Detected outliers using **IQR method**.
   - Capped extreme values to reduce skew and avoid model bias.
   - Credit score capped between 300 and 850.

4. **Standardization**:
   - Numeric features scaled using **StandardScaler**.

5. **Feature engineering**:
   - `total_expenses`, `savings_potential`, `dti`, `expense_ratio`, `affordability_ratio`, `salary_credit_interaction`, `emi_gap`, `balance_emi_gap`
   - Flags for missing financial data: `salary_missing`, `balance_missing`, `fund_missing`

6. **Encoding**:
   - One-hot encoding for categorical variables (e.g., gender, marital_status, education, etc.)

---

## Exploratory Data Analysis (EDA)
- Countplots for EMI eligibility across demographics and lending scenarios.
- Heatmap showing correlation between financial variables.
- Boxplots and statistical summaries to identify patterns and risk factors.
- Observed trends like higher credit score and higher savings potential increasing EMI eligibility.

---

## Machine Learning Models

### Classification (EMI Eligibility)
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Gradient Boosting Classifier**

**Metrics logged using MLflow**:
- Accuracy, Precision, Recall, F1-score, ROC-AUC

**Handling Imbalance**:
- SMOTE used to balance minority classes for improved classification.

---

### Regression (Maximum EMI)
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Gradient Boosting Regressor**

**Metrics logged using MLflow**:
- Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R², MAPE

---

## Model Deployment & MLflow
- Models logged, registered, and transitioned to **Production stage** using MLflow.
- **Label encoder** and **scaler** saved as artifacts for consistent predictions on new data.
- Prediction pipeline:
  1. Accept user input
  2. Compute engineered features
  3. Scale numeric values
  4. One-hot encode categorical variables
  5. Predict EMI eligibility (classification) and maximum EMI (regression)

---

## How to Use
1. Clone the repository:
```bash

https://github.com/Tasneem-Inayath/EMIPredict-AI---Intelligent-Financial-Risk-Assessment-Platform
