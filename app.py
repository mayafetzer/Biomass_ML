import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Page settings
st.set_page_config(page_title="Biomass ML Predictor", layout="wide")
st.title("Biomass ML Predictor")

# Define input features
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

# Pretty name → Actual file name
target_name_map = {
    "Absorption_Kinetics_PFO_Qexp(mg/g)": "Absorption_Kinetics_PFO_Qexp_mg_g__best_model.pkl",
    "Absorption_Kinetics_PFO_Qe cal(mg/g)": "Absorption_Kinetics_PFO_Qe cal_mg_g__best_model.pkl",
    "K1(min-1)": "K1_min-1__best_model.pkl",
    "Absorption_Kinetics_PSO_Qe cal(mg/g)": "Absorption_Kinetics_PSO_Qe cal_mg_g__best_model.pkl",
    "Absorption_Kinetics_PSO_K2(mg/g.min)": "Absorption_Kinetics_PSO_K2_mg_g_min__best_model.pkl",
    "Isotherm_Langmuir_Qmax(mg/g)": "Isotherm_Langmuir_Qmax_mg_g__best_model.pkl",
    "Isotherm_Langmuir_KL(L/mg)": "Isotherm_Langmuir_KL_L_mg__best_model.pkl",
    "Isotherm_Freundlich_Kf(mg/g)": "Isotherm_Freundlich_Kf_mg_g__best_model.pkl",
    "Isotherm_Freundlich_1/n": "Isotherm_Freundlich_1_n_best_model.pkl",  # No double underscore
    "PORE VOLUME(cm3/g)": "PORE VOLUME_cm3_g__best_model.pkl",
    "SURFACE AREA(m2/g)": "SURFACE AREA_m2_g__best_model.pkl",
    "ΔG(kJ /mol)": "ΔG_kJ _mol__best_model.pkl",
    "ΔH( kJ/mol)": "ΔH_ kJ_mol__best_model.pkl",
    "ΔS( J/mol)": "ΔS_ J_mol__best_model.pkl"
}

# Folder selection
model_folders = {
    "Pharma with Categorical Features": "pharma_categorical_models",
    "Pharma without Categorical Features": "pharma_no_categorical_models",
    "Dye with Categorical Features": "dye_categorical_models",
    "Dye without Categorical Features": "dye_no_categorical_models"
}

model_type = st.selectbox("Select Model Type", list(model_folders.keys()))
selected_folder = model_folders[model_type]
use_categorical = "no_categorical" not in selected_folder.lower()

# Input interface
st.subheader("Input Features")
user_input = {}

if use_categorical:
    for feature in categorical_features:
        user_input[feature] = st.text_input(feature)

for feature in numerical_features:
    user_input[feature] = st.number_input(feature, step=0.01, format="%.4f")

# Create input dataframe
input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict"):
    predictions = []
    missing_models = []

    for pretty_name, file_name in target_name_map.items():
        model_path = os.path.join(selected_folder, file_name)

        if not os.path.exists(model_path):
            missing_models.append(pretty_name)
            predictions.append(None)
            continue

        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            pred = model.predict(input_df)[0]
            predictions.append(pred)
        except Exception as e:
            st.error(f"Error predicting '{pretty_name}': {e}")
            predictions.append(None)

    # Output
    result_df = pd.DataFrame([predictions], columns=target_name_map.keys())
    st.subheader("Predicted Targets")
    st.write(result_df)

    if missing_models:
        st.warning(f"Missing models for: {', '.join(missing_models)}")
