import streamlit as st
import os
import pickle
import pandas as pd

# Set up Streamlit page
st.set_page_config(page_title="Biomass ML Predictor", layout="centered")
st.title("🌿 Biomass-Based Adsorption Predictor")

# Load available model files
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

selected_model_filename = st.selectbox("🔍 Choose a model", model_files)

if selected_model_filename:
    # Load selected model
    model_path = os.path.join(model_dir, selected_model_filename)
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Determine if categorical inputs should be used
    use_categoricals = "no_categorical" not in selected_model_filename.lower()

    st.header("🧪 Enter Feature Values")

    input_data = {}

    # Optional categorical inputs
    if use_categoricals:
        input_data["TYPE OF BIOMASS"] = st.selectbox("TYPE OF BIOMASS", ["Type1", "Type2", "Type3"])
        input_data["ADSORBENT"] = st.selectbox("ADSORBENT", ["A1", "A2", "A3"])
        input_data["ADSORBATE"] = st.selectbox("ADSORBATE", ["B1", "B2", "B3"])

    # Numerical inputs (explicitly cast as floats)
    input_data["MASS OF ADSORBENT(mg/L)"] = float(
        st.number_input("MASS OF ADSORBENT (mg/L)", value=10.0)
    )
    input_data["VOLUME OF DYE/POLLUTANT(mL)"] = float(
        st.number_input("VOLUME OF DYE/POLLUTANT (mL)", value=10.0)
    )
    input_data["Ph"] = float(st.number_input("pH", value=7.0))
    input_data["INITIAL CONCENTRATION OF ADSORBENT(mg/L)"] = float(
        st.number_input("INITIAL CONCENTRATION OF ADSORBENT (mg/L)", value=50.0)
    )
    input_data["CONTACT TIME(MIN)"] = float(
        st.number_input("CONTACT TIME (min)", value=60.0)
    )
    input_data["TEMPERATURE(K)"] = float(
        st.number_input("TEMPERATURE (K)", value=298.0)
    )

    # Prediction
    if st.button("📈 Predict"):
        input_df = pd.DataFrame([input_data])

        try:
            prediction = model.predict(input_df)

            # Format output as DataFrame if multi-output
            if hasattr(model, "feature_names_out_"):
                output_df = pd.DataFrame(prediction, columns=model.feature_names_out_)
            else:
                output_df = pd.DataFrame(prediction)

            st.success("✅ Prediction successful!")
            st.write("### 🔬 Predicted Outputs:")
            st.dataframe(output_df)

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
