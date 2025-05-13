import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="Biomass ML Predictor", layout="wide")

st.title("Biomass ML Predictor")

# Define categorical and numerical features
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

# Define all 14 target variable names
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

# Mapping of model types to folder names
model_folders = {
    "Pharma with Categorical Features": "pharma_categorical_models",
    "Pharma without Categorical Features": "pharma_no_categorical_models",
    "Dye with Categorical Features": "dye_categorical_models",
    "Dye without Categorical Features": "dye_no_categorical_models"
}

# Let user pick a model type
model_type = st.selectbox("Select Model Type", list(model_folders.keys()))
selected_folder = model_folders[model_type]

# Determine whether to show categorical inputs
use_categorical = "no_categorical" not in selected_folder.lower()

# Input form
st.subheader("Input Features")
user_input = {}

if use_categorical:
    for feature in categorical_features:
        user_input[feature] = st.text_input(f"{feature}")
        
for feature in numerical_features:
    user_input[feature] = st.number_input(f"{feature}", step=0.01, format="%.4f")

# Convert inputs to DataFrame
input_df = pd.DataFrame([user_input])

# Predict button
if st.button("Predict"):
    try:
        predictions = []

        for target in target_names:
            model_file = f"{target}.pkl"
            model_path = os.path.join(selected_folder, model_file)
            
            if not os.path.exists(model_path):
                st.warning(f"Model for '{target}' not found in {selected_folder}. Skipping.")
                predictions.append(None)
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            pred = model.predict(input_df)[0]
            predictions.append(pred)

        # Display predictions in table format
        result_df = pd.DataFrame([predictions], columns=target_names)
        st.subheader("Predicted Targets:")
        st.write(result_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
