pip install openpyxl
import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load model function
@st.cache
def load_model(model_filename):
    with open(model_filename, 'rb') as f:
        model = pickle.load(f)
    return model

# Load label encoder function (if you have categorical encoding)
@st.cache
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

    # Prepare data (Example: Encoding categorical features if needed)
    if 'Category' in data.columns:  # Adjust 'Category' to the actual column if applicable
        le = load_label_encoder('label_encoder.pkl')
        data['Category'] = le.transform(data['Category'])

    # Make predictions
    predictions = model.predict(data)
    st.write("Predictions:")
    st.write(predictions)
