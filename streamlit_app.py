import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, io

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="Disease Prediction",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------
# Load Model & Config
# ----------------------------
@st.cache_resource
def load_artifacts():
    """Load trained ML pipeline and config."""
    model = joblib.load("disease_pipeline.joblib")
    with open("config.json", "r") as f:
        config = json.load(f)
    return model, config

# ----------------------------
# Prediction Helper
# ----------------------------
def predict_df(model, df, features):
    """Run prediction and return (predictions, probabilities)."""
    X = df[features].copy()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba

# ----------------------------
# Single Input Form
# ----------------------------
def single_input_form(features, feature_ranges):
    """Render UI for manual single input prediction."""
    st.subheader("🔹 Single Prediction")
    values = {}
    cols = st.columns(2)

    for i, f in enumerate(features):
        rng = feature_ranges.get(f, {"min": 0.0, "max": 100.0, "median": 0.0})
        with cols[i % 2]:
            values[f] = st.number_input(
                f,
                value=float(rng["median"]),
                min_value=float(rng["min"]),
                max_value=float(rng["max"])
            )
    if st.button("Predict"):
        return pd.DataFrame([values])
    return None

# ----------------------------
# Batch Upload Section
# ----------------------------
def batch_upload_section(features):
    """Render UI for CSV batch prediction."""
    st.subheader("📂 Batch Prediction (CSV)")
    up = st.file_uploader("Upload CSV with same feature columns", type=["csv"])

    if up is not None:
        df = pd.read_csv(up)
        st.write("Preview:", df.head())

        # validate required columns
        missing = [c for c in features if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            return None
        return df
    return None

# ----------------------------
# Main App
# ----------------------------
def main():
    st.title("🩺 Disease Prediction App")
    st.write("This app predicts disease likelihood using a machine learning model trained in Colab.")

    # Load model + config
    model, config = load_artifacts()
    features = config["features"]
    feature_ranges = config.get("feature_ranges", {})
    class_names = config.get("class_names", ["Class 0", "Class 1"])

    # Mode selection
    mode = st.radio("Choose mode:", ["Single Prediction", "Batch Prediction"], horizontal=True)

    # Single prediction
    if mode == "Single Prediction":
        df_single = single_input_form(features, feature_ranges)
        if isinstance(df_single, pd.DataFrame):
            pred, proba = predict_df(model, df_single, features)
            st.success(f"Prediction: {class_names[int(pred[0])]}")
            st.metric("Probability of Positive class", f"{proba[0]:.3f}")

    # Batch prediction
    else:
        df_batch = batch_upload_section(features)
        if isinstance(df_batch, pd.DataFrame):
            pred, proba = predict_df(model, df_batch, features)
            out = df_batch.copy()
            out["Prediction"] = [class_names[p] for p in pred]
            out["Probability_Positive"] = proba

            st.write("Results Preview:", out.head())

            # Download button
            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_buf.getvalue(),
                file_name="predictions.csv",
                mime="text/csv"
            )

    # About Section
    with st.expander("ℹ️ About this project"):
        st.markdown("""
        - **Model**: Scikit-learn pipeline (imputation + scaling + SMOTE + RandomForest/LogReg)  
        - **Threshold**: 0.5 (can be tuned for precision/recall trade-off)  
        - **Data**: Kaggle Diabetes dataset  
        """)

# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    main()
