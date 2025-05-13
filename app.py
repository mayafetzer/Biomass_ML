import streamlit as st
import pickle
import pandas as pd
import os

# Title
st.title("Biomass Adsorption Predictor")

# Get all pickle files in current directory
model_files = [f for f in os.listdir() if f.endswith(".pkl")]

# Dropdown to choose a model
selected_model_file = st.selectbox("Select a model", model_files)

# Load the selected model
with open(selected_model_file, "rb") as f:
    model = pickle.load(f)

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

# Input form
with st.form("input_form"):
    st.subheader("Enter Feature Values")
    
    inputs = {}
    
    # Categorical inputs (only if model has them)
    if "no_categorical" not in selected_model_file.lower():
        for feature in categorical_features:
            inputs[feature] = st.text_input(feature)
    
    # Numerical inputs as floats
    for feature in numerical_features:
        value = st.text_input(feature)
        try:
            inputs[feature] = float(value)
        except ValueError:
            inputs[feature] = None
    
    submit = st.form_submit_button("Predict")

# Predict and display results
if submit:
    # Check for missing inputs
    if None in inputs.values():
        st.error("Please fill in all numerical fields with valid numbers.")
    else:
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)

        # Display each target's prediction
        st.subheader("Predicted Targets")
        for name, value in zip(target_names, prediction[0]):
            st.write(f"**{name}**: {value:.4f}")
