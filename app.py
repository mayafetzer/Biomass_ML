import pandas as pd
import pickle
import streamlit as st

# 1. Streamlit App Setup
st.title("Model Prediction App")
st.write("Select a model and upload a dataset (CSV or Excel) to make predictions.")

# 2. Model Selection Dropdown with Descriptive Titles
model_options = {
    'pharma_categorical.pkl': 'Pharmaceutical dataset with Categorical Variables',
    'pharma_no_categorical.pkl': 'Pharmaceutical dataset without Categorical Variables',
    'dye_categorical.pkl': 'Dye dataset with Categorical Variables',
    'dye_no_categorical.pkl': 'Dye dataset without Categorical Variables'
}

model_option = st.selectbox(
    "Choose a model to use for predictions:",
    list(model_options.keys()),
    format_func=lambda x: model_options[x]
)

# 3. File Uploader for Input Data (Accepting .csv and .xlsx)
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Check file extension and load accordingly
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    
    # Load the selected model
    with open(model_option, 'rb') as f:
        model = pickle.load(f)
    
    # Display model info to the user
    st.write(f"Using model: {model_options[model_option]}")
    
    # Assuming the model uses all columns except the target for prediction:
    feature_cols = [col for col in data.columns if col != 'Target']  # Modify 'Target' based on your dataset

    # Check if the necessary columns exist in the uploaded file
    missing_columns = [col for col in feature_cols if col not in data.columns]
    if missing_columns:
        st.error(f"Missing columns in the uploaded dataset: {', '.join(missing_columns)}")
    else:
        # Select the features for prediction
        X_new = data[feature_cols]

        # Make predictions with the model
        predictions = model.predict(X_new)

        # Display the predictions along with the original data
        data['Predictions'] = predictions
        st.write("Predictions:")
        st.write(data)
