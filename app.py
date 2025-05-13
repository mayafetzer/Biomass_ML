import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Title
st.title("Biomass ML Model Prediction")

# Model selection
model_files = [
    "pharma_categorical.pkl",
    "pharma_no_categorical.pkl",
    "dye_categorical.pkl",
    "dye_no_categorical.pkl"
]
model_choice = st.selectbox("Select a model file", model_files)

# Load model
model = None
try:
    with open(model_choice, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error(f"Model file '{model_choice}' not found.")
    st.stop()

# Feature inputs
st.subheader("Enter Input Features")

# Determine whether to show categorical features
show_categorical = "no_categorical" not in model_choice

input_data = {}

if show_categorical:
    input_data["TYPE OF BIOMASS"] = st.selectbox("TYPE OF BIOMASS", ["Type1", "Type2", "Type3"])
    input_data["ADSORBENT"] = st.selectbox("ADSORBENT", ["Adsorbent1", "Adsorbent2", "Adsorbent3"])
    input_data["ADSORBATE"] = st.selectbox("ADSORBATE", ["Adsorbate1", "Adsorbate2", "Adsorbate3"])

# Numerical features
numerical_features = [
    "MASS OF ADSORBENT(mg/L)",
    "VOLUME OF DYE/POLLUTANT(mL)",
    "Ph",
    "INITIAL CONCENTRATION OF ADSORBENT(mg/L)",
    "CONTACT TIME(MIN)",
    "TEMPERATURE(K)"
]

for feature in numerical_features:
    input_data[feature] = float(st.text_input(feature, value="0.0"))

# Make prediction
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    try:
        prediction = model.predict(input_df)
        st.subheader("Predicted Values")

        # Define target names for display
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

        if prediction.ndim == 1:
            prediction = prediction.reshape(1, -1)

        for name, value in zip(target_names, prediction[0]):
            st.write(f"**{name}**: {value:.4f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
