import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Feature Engineering", page_icon="⚙️", layout="wide")
st.title("⚙️ Feature Engineering & SMOTE Overview")

st.markdown("""
This page highlights the **engineered features** and how **SMOTE** handled class imbalance for EMI eligibility.
- Derived features improve model interpretability and performance
- SMOTE ensures fair representation of all classes during training
""")

# --- Load dataset ---
@st.cache_data
def load_data():
    clean_df = pd.read_csv("emi_prediction_dataset_final_enhanced.csv")  # includes engineered features
    smote_df = pd.read_csv("smote_emi_data.csv")    # SMOTE-applied training data
    return clean_df, smote_df

clean_df, smote_df = load_data()

# --- Derived Features Dictionary ---
derived_features = {
    "total_expenses": "Sum of school fees, college fees, travel, groceries, and other expenses",
    "dti": "Current EMI ÷ Monthly Salary",
    "expense_ratio": "Total Expenses ÷ Monthly Salary",
    "emi_gap": "Max Monthly EMI – Current EMI",
    "affordability_ratio": "Max EMI ÷ Required EMI",
    "savings_potential": "Bank Balance + Emergency Fund – Current EMI",
    "credit_risk": "Binary flag: risky if credit score < 600",
    "employment_stability": "Binary flag: stable if ≥5 years",
    "salary_credit_interaction": "Monthly Salary × Credit Score",
    "balance_emi_gap": "Bank Balance × EMI Gap"
}

# --- Feature Explanation Table ---
st.subheader("📘 Engineered Features Explained")
st.table(pd.DataFrame(list(derived_features.items()), columns=["Feature", "Description"]))

# --- Sample of Engineered Features ---
st.subheader("📊 Sample of Engineered Features")
available_cols = [col for col in derived_features.keys() if col in clean_df.columns]
if available_cols:
    st.dataframe(clean_df[available_cols].head(10), use_container_width=True)
else:
    st.warning("No engineered features found in the dataset.")

# --- Interactive Boxplot ---
if "emi_eligibility" in clean_df.columns and available_cols:
    st.subheader("📈 Engineered Feature vs EMI Eligibility")
    selected_eng = st.selectbox("Select a derived feature:", available_cols)

    fig, ax = plt.subplots()
    sns.boxplot(x="emi_eligibility", y=selected_eng, data=clean_df, palette="Set2", ax=ax)
    ax.set_title(f"{selected_eng} by EMI Eligibility")
    st.pyplot(fig)

# --- SMOTE Class Balance Visualization ---
if "emi_eligibility" in smote_df.columns:
    st.subheader("⚖️ EMI Eligibility Distribution After SMOTE")
    class_counts = smote_df["emi_eligibility"].value_counts()

    fig, ax = plt.subplots()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette="pastel", ax=ax)
    ax.set_ylabel("Count")
    ax.set_title("Class Balance After SMOTE")
    st.pyplot(fig)

    st.write("Class counts after SMOTE:")
    st.write(class_counts)

# Footer
st.markdown("---")
st.caption("Feature Engineering & SMOTE page | EMI Prediction App")