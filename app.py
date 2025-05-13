import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Set page config
st.set_page_config(page_title="Biomass ML Predictor", layout="wide")

st.title("Biomass ML Predictor")

# Define features
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

# Define target names and corresponding file-safe names
target_name_map = {
    "Absorption_Kinetics_PFO_Qexp(mg/g)": "Absorption_Kinetics_PFO_Qexp_mg_g_",
    "Absorption_Kinetics_PFO_Qe cal(mg/g)": "Absorption_Kinetics_PFO_Qe cal_mg_g_",
    "K1(min-1)": "K1_min_1_",
    "Absorption_Kinetics_PSO_Qe cal(mg/g)": "Absorption_Kinetics_PSO_Qe cal_mg_g_",
    "Absorption_Kinetics_PSO_K2(mg/g.min)": "Absorption_Kinetics_PSO_K2_mg_g_min_",
    "Isotherm_Langmuir_Qmax(mg/g)": "Isotherm_Langmuir_Qmax_mg_g_",
    "Isotherm_Langmuir_KL(L/mg)": "Isotherm_Langmuir_KL_L_mg_",
    "Isotherm_Freundlich_Kf(mg/g)": "Isotherm_Freundlich_Kf_mg_g_",
    "Isotherm_Freundlich_1/n": "Isotherm_Freundlich_1_n_",
    "PORE VOLUME(cm3/g)": "PORE VOLUME_cm3_g_",
    "SURFACE AREA(m2/g)": "SURFACE AREA_m2_g_",
    "ΔG(kJ /mol)": "ΔG_kJ_mol_",
    "ΔH( kJ/mol)": "ΔH_kJ_mol_",
    "ΔS( J/mol)": "ΔS_J_mol_"
}

# Let user pick model type
model_folders = {
    "Pharma with Categorical Features": "pharma_categorical_models",
    "Pharma without Categorical Features": "pharma_no_categorical_models",
    "Dye with Categorical Features": "dye_categorical_models",
    "Dye without Categorical Features": "dye_no_categorical_models"
}

model_type = st.selectbox("Select Model Type", list(model_folders.keys()))
selected_folder = model_folders[model_type]

# Determine if categorical inputs are needed
use_categorical = "no_categorical" not in selected_folder.lower()

# Input features
st.subheader("Input Features")
user_input = {}

if use_categorical:
    for feature in categorical_features:
        user_input[feature] = st.text_input(f"{feature}")

for feature in numerical_features:
    user_input[feature] = st.number_input(f"{feature}", step=0.01, format="%.4f")

input_df = pd.DataFrame([user_input])

# Predict
if st.button("Predict"):
    try:
        predictions = []

        for pretty_name, file_safe_prefix in target_name_map.items():
            # Compose filename
            model_file = f"{file_safe_prefix}best_model.pkl"
            model_path = os.path.join(selected_folder, model_file)

            if not os.path.exists(model_path):
                st.warning(f"Model for '{pretty_name}' not found. Skipping.")
                predictions.append(None)
                continue

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            pred = model.predict(input_df)[0]
            predictions.append(pred)

        # Display results
        result_df = pd.DataFrame([predictions], columns=target_name_map.keys())
        st.subheader("Predicted Targets:")
        st.write(result_df)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
