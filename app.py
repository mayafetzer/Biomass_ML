import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Define features
categorical_features = ["TYPE OF BIOMASS", "ADSORBENT", "ADSORBATE"]
numerical_features = [
    "MASS OF ADSORBENT(mg/L)",
    "VOLUME OF DYE/POLLUTANT(mL)",
    "Ph",
    "INITIAL CONCENTRATION OF ADSORBENT(mg/L)",
    "CONTACT TIME(MIN)",
    "TEMPERATURE(K)"
]
all_features = categorical_features + numerical_features

# Define target names
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

st.title("Biomass ML Predictor")

# Load models
model_dir = "."
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
selected_model_file = st.selectbox("Select a model", model_files)

# Show appropriate inputs
input_data = {}
use_categorical = "no_categorical" not in selected_model_file

st.subheader("Input Features")

if use_categorical:
    for feature in categorical_features:
        input_data[feature] = st.text_input(f"{feature}", "")
for feature in numerical_features:
    val = st.text_input(f"{feature}", "")
    try:
        input_data[feature] = float(val)
    except ValueError:
        input_data[feature] = None

# Make prediction
if st.button("Predict"):
    if None in input_data.values():
        st.error("Please fill in all fields with valid values.")
    else:
        # Create input DataFrame
        input_df = pd.DataFrame([input_data])
        # Load model
        with open(os.path.join(model_dir, selected_model_file), "rb") as f:
            model = pickle.load(f)
        # Predict
        predictions = model.predict(input_df)
        st.subheader("Predicted Values")
        for name, pred in zip(target_names, predictions[0]):
            st.write(f"**{name}**: {pred:.4f}")
