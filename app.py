import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# -------------------- LOAD SAVED ARTIFACTS --------------------
model = tf.keras.models.load_model("model.h5")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")  # <-- important for PCA consistency
feature_names = joblib.load("feature_names.pkl")

# -------------------- APP TITLE & INFO --------------------
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🧬", layout="wide")

st.title("🧬 Breast Cancer Detection App (ANN + PCA)")
st.write("""
This app predicts whether a breast tumor is **Malignant (M)** or **Benign (B)**  
based on diagnostic measurements from a cell nucleus.  
The model uses **Artificial Neural Networks (ANN)** and **PCA (Principal Component Analysis)** for dimensionality reduction.
""")

st.markdown("---")

# -------------------- INPUT FIELDS --------------------
st.subheader("🔢 Enter the feature values below:")

input_data = {}
cols = st.columns(3)  # three columns layout

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        input_data[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            value=0.0,
            format="%.4f"
        )

# -------------------- PREDICTION LOGIC --------------------
if st.button("🔮 Predict"):
    # Convert input to DataFrame
    new_sample = pd.DataFrame([input_data])
    new_sample = new_sample[feature_names]

    # Apply same preprocessing as training
    new_sample_scaled = scaler.transform(new_sample)
    new_sample_pca = pca.transform(new_sample_scaled)

    # Predict
    prediction = model.predict(new_sample_pca)
    pred_label = le.inverse_transform([int(prediction[0] > 0.5)])

    # -------------------- DISPLAY RESULT --------------------
    st.markdown("---")
    st.subheader("📊 Prediction Result:")

    if pred_label[0] == "M":
        st.error("⚠️ **Malignant (Cancerous)** detected.\nPlease consult a doctor immediately.")
    else:
        st.success("✅ **Benign (Non-Cancerous)**. No immediate concern detected.")

    st.caption(f"**Model Confidence:** {float(prediction[0]):.4f}")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit, TensorFlow, and scikit-learn.")
