import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import zscore

# Load model function
@st.cache_data
def load_model(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Load label encoder function (if you have categorical encoding)
@st.cache_data
def load_label_encoder(filename):
    with open(filename, 'rb') as f:
        le = pickle.load(f)
    return le

# Streamlit App Interface
st.title("Machine Learning Model Selector")

# File upload
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)

    st.write("Data Preview:")
    st.dataframe(data.head())

    # Drop unnecessary columns if they exist
    columns_to_drop = ['S/NO', 'Unnamed: 24']
    data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # Convert 'Ph' column to numeric, forcing errors to NaN (if any non-numeric values exist)
    data['Ph'] = pd.to_numeric(data['Ph'], errors='coerce')

    # One-hot encode the data
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_data = pd.DataFrame(encoder.fit_transform(data.select_dtypes(include=['object'])))
    encoded_data.columns = encoder.get_feature_names_out()

    # Concatenate encoded columns with numerical columns
    numerical_data = data.select_dtypes(exclude=['object']).reset_index(drop=True)
    data = pd.concat([numerical_data, encoded_data], axis=1)

    # Remove outliers using z-score
    z_scores = np.abs(zscore(data.select_dtypes(include=[np.number])))
    threshold = 3
    outliers = (z_scores > threshold).any(axis=1)
    data = data[~outliers]

    st.write("Processed Data:")
    st.dataframe(data.head())

    # Choose the model based on selection
    model_option = st.selectbox("Choose a Model", [
        'Pharmaceutical dataset with Categorical Variables',
        'Pharmaceutical dataset without Categorical Variables',
        'Dye dataset with Categorical Variables',
        'Dye dataset without Categorical Variables'
    ])

    # Dictionary of models
    model_files = {
        'Pharmaceutical dataset with Categorical Variables': 'pharma_categorical.pkl',
        'Pharmaceutical dataset without Categorical Variables': 'pharma_no_categorical.pkl',
        'Dye dataset with Categorical Variables': 'dye_categorical.pkl',
        'Dye dataset without Categorical Variables': 'dye_no_categorical.pkl'
    }

    # Load the selected model
    selected_model_file = model_files[model_option]
    model = load_model(selected_model_file)

    # Prepare data for prediction (Example: Encoding categorical features if needed)
    if 'Category' in data.columns:  # Adjust 'Category' to the actual column if applicable
        le = load_label_encoder('label_encoder.pkl')
        data['Category'] = le.transform(data['Category'])

    # Make predictions
    predictions = model.predict(data)
    st.write("Predictions:")
    st.write(predictions)
