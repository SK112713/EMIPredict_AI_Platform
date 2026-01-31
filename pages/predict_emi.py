import streamlit as st
import pandas as pd
import numpy as np
import mlflow.pyfunc
import joblib

# ------------------------
# Load trained artifacts
# ------------------------
trained_features = pd.read_csv("trained_features.csv")["feature"].tolist()
scaler = joblib.load("input_scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ------------------------
# Load MLflow models
# ------------------------
classification_model = mlflow.pyfunc.load_model("mlartifacts/924749176205125717/models/m-eca40b7e777b4d6e90c8b932547a17d6/artifacts")
regression_model = mlflow.pyfunc.load_model("mlartifacts/779327931942531374/models/m-55b8bd1e138141f18d7b10d87989c3b3/artifacts")

# ------------------------
# Helper function to compute features
# ------------------------
def compute_features(user_input: dict):
    features = user_input.copy()

    # Total expenses
    total_expenses = (
        features.get("school_fees", 0) + features.get("college_fees", 0) +
        features.get("travel_expenses", 0) + features.get("groceries_utilities", 0) +
        features.get("other_monthly_expenses", 0) + features.get("monthly_rent", 0)
    )
    features["total_expenses"] = total_expenses
    features["savings_potential"] = features.get("monthly_salary", 0) - total_expenses

    features["dti"] = features.get("current_emi_amount", 0) / max(features.get("monthly_salary", 1), 1)
    features["expense_ratio"] = total_expenses / max(features.get("monthly_salary", 1), 1)
    features["affordability_ratio"] = (
        (features.get("bank_balance", 0) + features.get("emergency_fund", 0)) /
        max(features.get("requested_amount", 1), 1)
    )
    features["salary_credit_interaction"] = features.get("monthly_salary", 0) * features.get("credit_score", 0)
    features["emi_gap"] = 0 - features.get("current_emi_amount", 0)
    features["balance_emi_gap"] = features.get("bank_balance", 0) - features.get("current_emi_amount", 0)

    # Missing flags
    features["salary_missing"] = int(features.get("monthly_salary", 0) == 0)
    features["balance_missing"] = int(features.get("bank_balance", 0) == 0)
    features["fund_missing"] = int(features.get("emergency_fund", 0) == 0)

    # One-hot encoding for categorical variables
    categorical_map = {
        "gender": ["Male"],
        "marital_status": ["Single"],
        "education": ["High School", "Post Graduate", "Professional"],
        "employment_type": ["Private", "Self-employed"],
        "company_type": ["MNC", "Mid-size", "Small", "Startup"],
        "house_type": ["Own", "Rented"],
        "existing_loans": ["Yes"],
        "emi_scenario": ["Education EMI", "Home Appliances EMI", "Personal Loan EMI", "Vehicle EMI"]
    }

    for cat_col, cat_values in categorical_map.items():
        for val in cat_values:
            features[f"{cat_col}_{val}"] = int(features.get(cat_col, "") == val)

    return features

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="EMI Eligibility Predictor", layout="wide")
st.title("💰 EMI Eligibility & Max EMI Predictor")

with st.form("user_input_form"):
    # ------------------------
    # Personal & Employment
    # ------------------------
    st.subheader("Personal & Employment Details")
    age = st.number_input("Age", min_value=18, max_value=80, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Married", "Single"])
    education = st.selectbox("Education", ["Graduate", "High School", "Post Graduate", "Professional"])
    employment_type = st.selectbox("Employment Type", ["Government", "Private", "Self-employed"])
    company_type = st.selectbox("Company Type", ["Large Indian", "MNC", "Mid-size", "Small", "Startup"])
    house_type = st.selectbox("House Type", ["Family", "Own", "Rented"])
    existing_loans = st.selectbox("Existing Loans", ["Yes", "No"])

    # ------------------------
    # Expenses Group
    # ------------------------
    st.subheader("Monthly Expenses")
    monthly_salary = st.number_input("Monthly Salary", min_value=0.0, step=100.0)
    monthly_rent = st.number_input("Monthly Rent", min_value=0.0, step=100.0)
    school_fees = st.number_input("School Fees", min_value=0.0, step=100.0)
    college_fees = st.number_input("College Fees", min_value=0.0, step=100.0)
    travel_expenses = st.number_input("Travel Expenses", min_value=0.0, step=100.0)
    groceries_utilities = st.number_input("Groceries & Utilities", min_value=0.0, step=100.0)
    other_monthly_expenses = st.number_input("Other Monthly Expenses", min_value=0.0, step=100.0)
    current_emi_amount = st.number_input("Current EMI Amount", min_value=0.0, step=100.0)

    # ------------------------
    # Financials
    # ------------------------
    st.subheader("Financial Information")
    credit_score = st.number_input("Credit Score", min_value=300, max_value=900, step=1)
    bank_balance = st.number_input("Bank Balance", min_value=0.0, step=100.0)
    emergency_fund = st.number_input("Emergency Fund", min_value=0.0, step=100.0)
    requested_amount = st.number_input("Requested Loan Amount", min_value=1000.0, step=1000.0)
    requested_tenure = st.number_input("Requested Tenure (months)", min_value=6, max_value=120, step=6)
    emi_scenario = st.selectbox("EMI Scenario", [
        "E-commerce Shopping EMI", "Education EMI", "Home Appliances EMI",
        "Personal Loan EMI", "Vehicle EMI"
    ])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Collect user input into dict
    user_input = {
        "age": age,
        "gender": gender,
        "marital_status": marital_status,
        "education": education,
        "employment_type": employment_type,
        "company_type": company_type,
        "house_type": house_type,
        "existing_loans": existing_loans,
        "monthly_salary": monthly_salary,
        "monthly_rent": monthly_rent,
        "school_fees": school_fees,
        "college_fees": college_fees,
        "travel_expenses": travel_expenses,
        "groceries_utilities": groceries_utilities,
        "other_monthly_expenses": other_monthly_expenses,
        "current_emi_amount": current_emi_amount,
        "credit_score": credit_score,
        "bank_balance": bank_balance,
        "emergency_fund": emergency_fund,
        "requested_amount": requested_amount,
        "requested_tenure": requested_tenure,
        "emi_scenario": emi_scenario
    }

    # Compute features
    features = compute_features(user_input)
    features_df = pd.DataFrame([features])
    features_df = features_df.reindex(columns=trained_features, fill_value=0)
    
    # Scale numeric columns
    numeric_cols = [
        'monthly_salary','monthly_rent','school_fees','college_fees','travel_expenses',
        'groceries_utilities','other_monthly_expenses','current_emi_amount','credit_score',
        'bank_balance','emergency_fund','requested_amount','requested_tenure','savings_potential',
        'dti','total_expenses','expense_ratio','affordability_ratio','salary_credit_interaction',
        'emi_gap','balance_emi_gap'
    ]
    features_df[numeric_cols] = scaler.transform(features_df[numeric_cols])

    # Make predictions
    pred_class_encoded = classification_model.predict(features_df)
    pred_class = label_encoder.inverse_transform(pred_class_encoded)
    pred_emi = regression_model.predict(features_df)

    # Display results
    st.subheader("💡 Prediction Results")
    st.write(f"**EMI Eligibility:** {pred_class[0]}")
    st.write(f"**Max EMI Amount:** ₹{pred_emi[0]:,.2f}")
