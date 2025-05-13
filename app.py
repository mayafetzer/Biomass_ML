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

# Define target names based on the provided list
target_names = [
    "Absorption_Kinetics_PFO_Qexp(mg/g)",
    "Absorption_Kinetics_PFO_Qe cal(mg/g)",
    "K1(min-1)",
    "Absorption_Kinetics_PSO_Qe cal(mg/g)",
    "Absorption_Kinetics_PSO_K2(mg/g.min)",
    "Isotherm_Langmuir_Qmax(mg/g)",
    "Isotherm_Langmuir_KL(L/mg)",
    "Isotherm_Freundlich_Kf(mg/g)",
    "Isotherm_Freundlich_1/n",
    "PORE VOLUME(cm3/g)",
    "SURFACE AREA(m2/g)",
    "ΔG(kJ /mol)",
    "ΔH( kJ/mol)",
    "ΔS( J/mol)"
]

# Predict
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        
        # Ensure prediction is in the right format (2D array for multiple targets)
        if isinstance(prediction, np.ndarray) and prediction.ndim == 2:
            # If multiple targets are predicted, align predictions with target names
            prediction_df = pd.DataFrame(prediction, columns=target_names[:prediction.shape[1]])
        else:
            # If only one target is predicted, display it in the correct format
            prediction_df = pd.DataFrame([prediction], columns=target_names[:len(prediction)])

        st.subheader("Predicted Targets:")
        st.write(prediction_df)
        
    except Exception as e:
        st.error(f"Error during prediction: {e}")
