import streamlit as st
import os
import pickle
import numpy as np

# Categorical and numerical feature names
categorical_features = ["TYPE OF BIOMASS", "ADSORBENT", "ADSORBATE"]
numerical_features = [
    "MASS OF ADSORBENT(mg/L)",
    "VOLUME OF DYE/POLLUTANT(mL)",
    "Ph",
    "INITIAL CONCENTRATION OF ADSORBENT(mg/L)",
    "CONTACT TIME(MIN)",
    "TEMPERATURE(K)"
]

# Target labels for display
target_outputs = [
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

# List all pickle model files in current directory
model_files = [f for f in os.listdir() if f.endswith('.pkl')]
if not model_files:
    st.warning("No model files (.pkl) found.")
else:
    model_choice = st.selectbox("Choose a model file", model_files)

    # Collect categorical inputs
    st.subheader("Categorical Features")
    cat_inputs = [st.text_input(f) for f in categorical_features]

    # Collect numerical inputs
    st.subheader("Numerical Features")
    num_inputs = [st.number_input(f, format="%.4f") for f in numerical_features]

    if st.button("Predict"):
        try:
            # Load selected model
            with open(model_choice, 'rb') as f:
                model = pickle.load(f)

            # Combine inputs
            input_data = np.array(cat_inputs + num_inputs, dtype=object).reshape(1, -1)

            # Run prediction
            prediction = model.predict(input_data)

            # Display results
            st.subheader("Predicted Outputs:")
            for label, value in zip(target_outputs, prediction[0]):
                st.write(f"**{label}**: {value:.4f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
