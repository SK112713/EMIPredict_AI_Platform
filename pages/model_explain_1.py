import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# ------------------------
# Page Config
# ------------------------
st.set_page_config(page_title="Model Explainability", page_icon="🔍", layout="wide")
st.title("🔍 Model Explainability & Insights")

# ------------------------
# Load Dataset
# ------------------------
@st.cache_data
def load_data():
    return pd.read_csv("smote_emi_data.csv")

df = load_data()

# ------------------------
# Load Training Feature Order
# ------------------------
trained_features = pd.read_csv("trained_features.csv")["feature"].tolist()

st.success("✅ Trained feature list loaded")
st.write("Expected feature count:", len(trained_features))

# ------------------------
# Align dataset with training schema
# ------------------------
X = df.reindex(columns=trained_features, fill_value=0)

# ------------------------
# Load MLflow Models
# ------------------------
classifier, regressor = None, None

try:
    classifier = mlflow.pyfunc.load_model(r"mlartifacts\924749176205125717\models\m-54a470dbc0e74d96803fb11bb04aa40d\artifacts")
    st.success("✅ Classifier Loaded")
except Exception as e:
    st.error(f"❌ Classifier load failed: {e}")

try:
    regressor = mlflow.pyfunc.load_model(r"mlartifacts\779327931942531374\models\m-2f6f829b3ad74473904abb4d6ad82a5e\artifacts")
    st.success("✅ Regressor Loaded")
except Exception as e:
    st.error(f"❌ Regressor load failed: {e}")

# ------------------------
# Display MLflow Metrics
# ------------------------
st.subheader("📊 Versioned Model Metrics")
import mlflow
import os

# Point to the parent directory where 'mlartifacts' and 'mlruns' exist
# If they are in your current project folder, use the current directory path
tracking_path = os.path.abspath("mlruns") 
mlflow.set_tracking_uri(f"file:///{os.path.dirname(tracking_path)}")

client = MlflowClient()

def show_metrics_by_id(run_id, label):
    client = MlflowClient()
    try:
        run = client.get_run(run_id)
        metrics = run.data.metrics
        params = run.data.params

        st.markdown(f"### {label}")
        st.json(metrics)
        st.json(params)
        # ... rest of your plotting code ...
    except Exception as e:
        st.error(f"Could not find metadata for {label} (Run ID: {run_id}): {e}")

# Extract these IDs from your local folders
show_metrics_by_id("692a4ec75f0f4850a382e0a8885afbab", "EMI Classifier")
show_metrics_by_id("729f90a441254754ada71b3b66d64745", "EMI Regressor")

# client = MlflowClient()

# def show_metrics(model_name, label):
#     versions = client.search_model_versions(f"name='{model_name}'")
#     prod_models = [v for v in versions if v.current_stage == "Production"]

#     if not prod_models:
#         st.warning(f"No production model found for {label}")
#         return

#     run_id = prod_models[0].run_id
#     run = client.get_run(run_id)
#     metrics = run.data.metrics
#     params = run.data.params

#     st.markdown(f"### {label}")
#     st.write("Version:", prod_models[0].version)
#     st.write("Run ID:", run_id)
#     st.json(metrics)
#     st.json(params)

#     if metrics:
#         fig, ax = plt.subplots()
#         ax.bar(metrics.keys(), metrics.values())
#         ax.set_title(f"{label} Metrics")
#         plt.xticks(rotation=45)
#         st.pyplot(fig)

# show_metrics("XGBoost_Classifier", "EMI Classifier")
# show_metrics("XGBoost_Regressor", "EMI Regressor")

# ------------------------
# Predictions Distribution
# ------------------------
st.subheader("📈 Predictions Distribution")

if classifier is not None:
    try:
        preds = classifier.predict(X)
        fig, ax = plt.subplots()
        pd.Series(preds).astype(str).value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Classifier Predictions")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Classifier Prediction Failed: {e}")

if regressor is not None:
    try:
        preds = regressor.predict(X)
        fig, ax = plt.subplots()
        pd.Series(preds).hist(bins=25, ax=ax)
        ax.set_title("Regressor Predictions")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Regressor Prediction Failed: {e}")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("EMI Explainability Page ✅")
