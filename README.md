# Biomass ML Predictor

This Streamlit app allows users to predict 14 scientific adsorption-related target variables using trained machine learning models based on user input features. The models are organized into four folders depending on the dataset (pharmaceutical or dye) and whether categorical features are used.

---

## Live Demo

You can access the live GUI here:  
**https://biomassml.streamlit.app/**

---

## Model Folders

Each folder contains 14 pickle files (one per target variable), with filenames like:

Absorption_Kinetics_PFO_Qexp_mg_g__best_model.pkl

K1_min-1__best_model.pkl

ΔG_kJ _mol__best_model.pkl

...


Available folders:

- `pharma_categorical_models`
- `pharma_no_categorical_models`
- `dye_categorical_models`
- `dye_no_categorical_models`

**Note**: These folders must be in the same directory as `app.py`.

---

## Input Features

### Categorical (used only for models with categorical data)

- TYPE OF BIOMASS  
- ADSORBENT  
- ADSORBATE

### Numerical

- MASS OF ADSORBENT (mg/L)  
- VOLUME OF DYE/POLLUTANT (mL)  
- Ph  
- INITIAL CONCENTRATION OF ADSORBENT (mg/L)  
- CONTACT TIME (MIN)  
- TEMPERATURE (K)

---

## Target Variables Predicted

- Absorption_Kinetics_PFO_Qexp (mg/g)  
- Absorption_Kinetics_PFO_Qe cal (mg/g)  
- K1 (min⁻¹)  
- Absorption_Kinetics_PSO_Qe cal (mg/g)  
- Absorption_Kinetics_PSO_K2 (mg/g·min)  
- Isotherm_Langmuir_Qmax (mg/g)  
- Isotherm_Langmuir_KL (L/mg)  
- Isotherm_Freundlich_Kf (mg/g)  
- Isotherm_Freundlich_1/n  
- PORE VOLUME (cm³/g)  
- SURFACE AREA (m²/g)  
- ΔG (kJ/mol)  
- ΔH (kJ/mol)  
- ΔS (J/mol)

---

## How to Run

1. Install dependencies:

    ```
    pip install streamlit pandas numpy
    ```

2. Run the app:

    ```
    streamlit run streamlit_app.py
    ```

3. Open the app in your browser (usually at `http://localhost:8501`).

---

## Notes

- The app will warn you if any model files are missing.
- All predictions are based on individual models trained for each target.
- Easily extendable for CSV upload, data export, or multi-sample prediction.

---

## License

This project is provided for research and educational purposes. Attribution is appreciated if used or modified.
