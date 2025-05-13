import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="Biomass ML Predictor", layout="wide")

st.title("Biomass ML Predictor")

# Define feature sets
categorical_features = [
    "TYPE OF BIOMASS",
    "ADSORBENT",
    "ADSORBATE"
]

numerical_features = [
    "MASS OF ADSORBENT(mg/L)",
    "VOLUME OF DYE/POLLUTANT(mL)",
    "Ph",
    "INITIAL CONCENTRATION OF ADSORBENT(mg/L)",
    "CONTACT TIME(MIN)",
    "TEMPERATURE(K)"
]

# Load pickle models
model_dir = "./models"  # Adjust if different
model_files = [
    "pharma_categorical.pkl",
    "pharma_no_categorical.pkl",
    "dye_categorical.pkl",
    "dye_no_categorical.pkl"
]

# Let user pick a model
model_choice = st.selectbox("Select a model", model_files)
model_path = os.path.join(model_dir, model_choice)

# Load the selected model
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Check if categorical inputs should be hidden
use_categorical = "no_categorical" not in model_choice.lower()

# Input UI
user_input = {}

st.subheader("Input Features")

if use_categorical:
    for feature in categorical_features:
        user_input[feature] = st.text_input(f"{feature}")

for feature in numerical_features:
    user_input[feature] = st.number_input(f"{feature}", step=0.01, format="%.4f")

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        st.subheader("Predicted Targets:")
        st.write(pd.DataFrame(prediction, columns=model.feature_names_out_ if hasattr(model, "feature_names_out_") else None))
    except Exception as e:
        st.error(f"Error during prediction: {e}")
