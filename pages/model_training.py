import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="Model Training & Comparison", page_icon="🤖", layout="wide")
st.title("🤖 Model Training & Comparison")

st.markdown("""
This page summarizes the performance of different machine learning models  
for **EMI eligibility classification** and **maximum EMI regression**,  
pulled directly from the local `mlruns` folder.
""")

# --- Helper: Load metrics and params from mlruns ---
import yaml

def load_mlruns(experiment_id):
    base_path = f"mlruns/{experiment_id}"
    records = []

    if not os.path.exists(base_path):
        return pd.DataFrame()

    for run_id in os.listdir(base_path):
        run_path = os.path.join(base_path, run_id)
        if not os.path.isdir(run_path):
            continue

        metrics_dir = os.path.join(run_path, "metrics")
        params_dir = os.path.join(run_path, "params")
        meta_path = os.path.join(run_path, "meta.yaml")

        if not os.path.exists(metrics_dir):
            continue

        # Read metrics
        metrics = {}
        for fname in os.listdir(metrics_dir):
            with open(os.path.join(metrics_dir, fname)) as f:
                line = f.readline().strip()
                if line:
                    metrics[fname] = float(line.split()[1])

        # Read params
        params = {}
        if os.path.exists(params_dir):
            for fname in os.listdir(params_dir):
                with open(os.path.join(params_dir, fname)) as f:
                    params[fname] = f.readline().strip()

        # Read run_name from meta.yaml
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = yaml.safe_load(f)
                run_name = meta.get("run_name", run_id)
        else:
            run_name = run_id

        records.append({
            "Run ID": run_id,
            "Model": run_name,
            **metrics
        })

    return pd.DataFrame(records)

# --- Classification Results ---
st.subheader("📊 Classification Models (Eligibility Prediction)")
classification_results = load_mlruns("924749176205125717")  # your classification experiment ID

if not classification_results.empty:
    st.dataframe(classification_results, use_container_width=True)

    metrics_to_plot = [m for m in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
                       if m in classification_results.columns]
    if metrics_to_plot:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=classification_results.melt(id_vars="Model", value_vars=metrics_to_plot,
                                             var_name="Metric", value_name="Score"),
            x="Model", y="Score", hue="Metric", ax=ax
        )
        ax.set_title("Classification Model Performance")
        plt.xticks(rotation=20)
        st.pyplot(fig)
else:
    st.warning("No classification runs found in mlruns.")

# --- Regression Results ---
# --- Regression Results ---
st.subheader("📈 Regression Models (Max EMI Prediction)")
regression_results = load_mlruns("779327931942531374")  # your regression experiment ID

if not regression_results.empty:
    st.dataframe(regression_results, use_container_width=True)

    # Plot RMSE, MAE, R² separately
    core_metrics = [m for m in ["mse", "rmse", "mae", "r2"] if m in regression_results.columns]
    if core_metrics:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=regression_results.melt(id_vars="Model", value_vars=core_metrics,
                                         var_name="Metric", value_name="Score"),
            x="Model", y="Score", hue="Metric", ax=ax1
        )
        ax1.set_title("Regression Metrics (excluding MAPE)")
        plt.xticks(rotation=20)
        st.pyplot(fig1)

    # Plot MAPE separately
    if "mape" in regression_results.columns:
        st.subheader("📉 MAPE (%) Comparison")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        sns.barplot(data=regression_results, x="Model", y="mape", palette="coolwarm", ax=ax2)
        ax2.set_title("Mean Absolute Percentage Error (MAPE)")
        ax2.set_ylabel("MAPE (%)")
        plt.xticks(rotation=20)
        st.pyplot(fig2)
else:
    st.warning("No regression runs found in mlruns.")
# --- Final Model Selection ---
st.subheader("✅ Final Model Selection")

# Select best classification model (highest F1 or ROC-AUC)
if not classification_results.empty:
    if "f1_score" in classification_results.columns:
        best_clf = classification_results.loc[classification_results["f1_score"].idxmax()]
    elif "roc_auc" in classification_results.columns:
        best_clf = classification_results.loc[classification_results["roc_auc"].idxmax()]
    else:
        best_clf = classification_results.iloc[0]

    st.success(f"Best Classification Model → **{best_clf['Model']}** "
               f"(F1: {best_clf.get('f1_score','N/A')}, ROC-AUC: {best_clf.get('roc_auc','N/A')})")

# Select best regression model (lowest RMSE)
if not regression_results.empty:
    if "rmse" in regression_results.columns:
        best_reg = regression_results.loc[regression_results["rmse"].idxmin()]
    else:
        best_reg = regression_results.iloc[0]

    st.success(f"Best Regression Model → **{best_reg['Model']}** "
               f"(RMSE: {best_reg.get('rmse','N/A')}, R²: {best_reg.get('r2','N/A')})")
# Footer
st.markdown("---")
st.caption("Model Training & Comparison page | EMI Prediction App")