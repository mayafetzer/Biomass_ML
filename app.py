import streamlit as st
import pickle
import pandas as pd
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

# Input fields for categorical features (user types in the values)
type_of_biomass = st.text_input('Enter TYPE OF BIOMASS')
adsorbent = st.text_input('Enter ADSORBENT')
adsorbate = st.text_input('Enter ADSORBATE')

# Input fields for numerical features
mass_of_adsorbent = st.number_input('Enter MASS OF ADSORBENT (mg/L)', min_value=0.0, step=0.1)
volume_of_dye_pollutant = st.number_input('Enter VOLUME OF DYE/POLLUTANT (mL)', min_value=0.0, step=0.1)
ph = st.number_input('Enter Ph', min_value=0.0, step=0.1)
initial_concentration_of_adsorbent = st.number_input('Enter INITIAL CONCENTRATION OF ADSORBENT (mg/L)', min_value=0.0, step=0.1)
contact_time = st.number_input('Enter CONTACT TIME (MIN)', min_value=0, step=1)
temperature = st.number_input('Enter TEMPERATURE (K)', min_value=0.0, step=0.1)

# Collect the user input into a data structure (e.g., a dictionary or DataFrame)
user_input = {
    'TYPE OF BIOMASS': [type_of_biomass],
    'ADSORBENT': [adsorbent],
    'ADSORBATE': [adsorbate],
    'MASS OF ADSORBENT (mg/L)': [mass_of_adsorbent],
    'VOLUME OF DYE/POLLUTANT (mL)': [volume_of_dye_pollutant],
    'Ph': [ph],
    'INITIAL CONCENTRATION OF ADSORBENT (mg/L)': [initial_concentration_of_adsorbent],
    'CONTACT TIME (MIN)': [contact_time],
    'TEMPERATURE (K)': [temperature]
}

# Convert the user input into a DataFrame
data = pd.DataFrame(user_input)

st.write("User Input Data:")
st.dataframe(data)

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
if 'TYPE OF BIOMASS' in data.columns:  # Adjust 'TYPE OF BIOMASS' to the actual column if applicable
    le = load_label_encoder('label_encoder.pkl')
    data['TYPE OF BIOMASS'] = le.transform(data['TYPE OF BIOMASS'])
    data['ADSORBENT'] = le.transform(data['ADSORBENT'])
    data['ADSORBATE'] = le.transform(data['ADSORBATE'])

# Make predictions
predictions = model.predict(data)
st.write("Predictions:")
st.write(predictions)
