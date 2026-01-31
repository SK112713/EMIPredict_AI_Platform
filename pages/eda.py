import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="EDA - EMI Prediction", page_icon="🔍", layout="wide")
st.title("🔍 Exploratory Data Analysis (EDA)")

st.markdown("""
This page provides an **exploratory analysis** of the EMI dataset:
- EMI eligibility distribution across scenarios
- Financial risk relationships
- Demographic patterns
- Statistical summaries and boxplots
""")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("emi_prediction_dataset_final_enhanced.csv")

df = load_data()

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# -----------------------------
# 1. EMI Eligibility Distribution Across Lending Scenarios
# -----------------------------
st.subheader("📊 EMI Eligibility Across EMI Types")

if 'emi_scenario' in df.columns:
    fig1, ax1 = plt.subplots()
    sns.countplot(x='emi_scenario', hue='emi_eligibility', data=df, ax=ax1)
    ax1.set_title("EMI Scenario vs Eligibility")
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
    st.pyplot(fig1)
else:
    st.warning("Column 'emi_scenario' not found in dataset.")

# -----------------------------
# 2. Correlation Heatmap
# -----------------------------
st.subheader("📈 Financial Correlation Matrix")

financial_vars = [
    'monthly_salary','dti','expense_ratio','savings_potential',
    'affordability_ratio','credit_score','bank_balance','emergency_fund'
]

existing_financial = [col for col in financial_vars if col in df.columns]

if len(existing_financial) >= 2:
    corr_matrix = df[existing_financial].corr()
    fig2, ax2 = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax2)
    ax2.set_title("Financial Feature Correlation")
    st.pyplot(fig2)
else:
    st.warning("Not enough financial columns for correlation heatmap.")

# -----------------------------
# 3. Demographic Analysis
# -----------------------------
st.subheader("👥 Demographic vs EMI Eligibility")

demographic_cols = [
    'gender','marital_status','education',
    'employment_type','company_type',
    'house_type','existing_loans'
]

for col in demographic_cols:
    if col in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=col, hue='emi_eligibility', data=df, ax=ax)
        ax.set_title(f"{col} vs EMI Eligibility")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)

# -----------------------------
# 4. Risk Flag Analysis (Only columns that exist)
# -----------------------------
st.subheader("⚠️ Risk Indicators")

risk_flags = [
    'salary_missing',
    'balance_missing',
    'fund_missing'
]

for flag in risk_flags:
    if flag in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(x=flag, hue='emi_eligibility', data=df, ax=ax)
        ax.set_title(f"{flag} vs EMI Eligibility")
        st.pyplot(fig)

# -----------------------------
# 5. Statistical Summary
# -----------------------------
st.subheader("📊 Statistical Summary by Eligibility")

summary_cols = [
    'monthly_salary','dti','expense_ratio',
    'savings_potential','affordability_ratio',
    'credit_score','bank_balance'
]

existing_summary = [col for col in summary_cols if col in df.columns]

summary = df.groupby('emi_eligibility')[existing_summary].describe()
st.dataframe(summary)

# -----------------------------
# 6. Boxplots
# -----------------------------
st.subheader("📦 Financial Feature Distributions")

for col in existing_summary:
    fig, ax = plt.subplots()
    sns.boxplot(x='emi_eligibility', y=col, data=df, ax=ax)
    ax.set_title(f"{col} vs EMI Eligibility")
    st.pyplot(fig)

# -----------------------------
# Footer
# -----------------------------
st.caption("EDA Page | EMI Risk Analytics Dashboard")
