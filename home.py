import streamlit as st

# Page config
st.set_page_config(page_title="EMI Prediction App", page_icon="💸", layout="centered")

# Title and intro
st.title("💸 EMI Prediction & Risk Assessment")
st.markdown("""
Welcome to the EMI Prediction App — a smart tool for evaluating loan eligibility and estimating monthly EMI capacity.

This app uses machine learning models trained on real financial data to help:
- Classify applicants into **Eligible**, **High Risk**, or **Not Eligible**
- Predict the **maximum EMI** an applicant can afford
- Provide insights into features and model performance
""")

# Navigation buttons
st.markdown("### 🔍 Navigate to:")

col1, = st.columns(1)  # Single column layout

with col1:
    if st.button("📊 Data Overview"):
        st.switch_page("pages/data_overview.py")

    if st.button("📈 Exploratory Data Analysis"):
        st.switch_page("pages/eda.py")

    if st.button("⚙️ Feature Engineering"):
        st.switch_page("pages/feature_eng.py")

    if st.button("🤖 Model Training & Comparison"):
        st.switch_page("pages/model_training.py")

    if st.button("🔮 Real-Time Prediction"):
        st.switch_page("pages/prediction_emi.py")

    if st.button("📘 MLflow Tracking"):
        st.switch_page("pages/model_explain.py")

# Optional image or logo
# st.image("logo.png", width=200)

# Footer
st.markdown("---")
st.caption("Built with ❤️ by Sarath Kumar R | Powered by MLflow + Streamlit")