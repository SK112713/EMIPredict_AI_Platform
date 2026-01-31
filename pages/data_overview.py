import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")
st.title("📊 Data Overview")

st.markdown("""
This page provides a quick overview of both **raw** and **cleaned** EMI datasets:
- Dataset shape and column details
- Summary statistics
- Class balance for **EMI Eligibility**
""")

# --- Load datasets ---
@st.cache_data
def load_datasets():
    raw_df = pd.read_csv("emi_prediction_dataset.csv")      # adjust path
    clean_df = pd.read_csv("emi_prediction_dataset_enhanced.csv") # adjust path
    return raw_df, clean_df

raw_df, clean_df = load_datasets()

# --- Dataset Shapes ---
st.subheader("📐 Dataset Shapes")
col1, col2 = st.columns(2)
with col1:
    st.write("**Raw Dataset**")
    st.write(f"Rows: {raw_df.shape[0]}, Columns: {raw_df.shape[1]}")
with col2:
    st.write("**Cleaned Dataset**")
    st.write(f"Rows: {clean_df.shape[0]}, Columns: {clean_df.shape[1]}")

# --- Column Info ---
st.subheader("🧾 Column Information")
col1, col2 = st.columns(2)
with col1:
    st.write("**Raw Dataset**")
    raw_col_info = pd.DataFrame({
        "Column": raw_df.columns,
        "Data Type": raw_df.dtypes.astype(str),
        "Missing Values": raw_df.isnull().sum().values
    })
    st.dataframe(raw_col_info, use_container_width=True)

with col2:
    st.write("**Cleaned Dataset**")
    clean_col_info = pd.DataFrame({
        "Column": clean_df.columns,
        "Data Type": clean_df.dtypes.astype(str),
        "Missing Values": clean_df.isnull().sum().values
    })
    st.dataframe(clean_col_info, use_container_width=True)

# --- Summary Statistics ---
st.subheader("📊 Summary Statistics")
col1, col2 = st.columns(2)
with col1:
    st.write("**Raw Dataset**")
    st.dataframe(raw_df.describe(include="all").transpose(), use_container_width=True)
with col2:
    st.write("**Cleaned Dataset**")
    st.dataframe(clean_df.describe(include="all").transpose(), use_container_width=True)

# --- Class Balance Comparison ---
if "emi_eligibility" in raw_df.columns and "emi_eligibility" in clean_df.columns:
    st.subheader("⚖️ EMI Eligibility Distribution Comparison")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Raw Dataset**")
        raw_counts = raw_df["emi_eligibility"].value_counts()
        fig1, ax1 = plt.subplots()
        raw_counts.plot(kind="bar", color=["gray", "orange", "red"], ax=ax1)
        ax1.set_ylabel("Count")
        ax1.set_title("Raw EMI Eligibility")
        st.pyplot(fig1)
        st.write(raw_counts)

    with col2:
        st.write("**Cleaned Dataset**")
        clean_counts = clean_df["emi_eligibility"].value_counts()
        fig2, ax2 = plt.subplots()
        clean_counts.plot(kind="bar", color=["green", "orange", "red"], ax=ax2)
        ax2.set_ylabel("Count")
        ax2.set_title("Cleaned EMI Eligibility")
        st.pyplot(fig2)
        st.write(clean_counts)

# Footer
st.markdown("---")
st.caption("Data Overview page | EMI Prediction App")