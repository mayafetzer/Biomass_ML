import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# User uploads a file
uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Try to read the file based on the extension
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a CSV or Excel file.")
    
    # Display the first few rows of the dataset
    st.write("Dataset preview:")
    st.write(df.head())

    # User option for handling categorical variables
    drop_categorical = st.checkbox('Drop Categorical Variables', value=True)

    # Handle categorical variables based on user selection
    if drop_categorical:
        # Drop categorical columns if the user chooses to
        categorical_cols = df.select_dtypes(include=['object']).columns
        df = df.drop(columns=categorical_cols)

    # Define target columns and feature columns
    target_cols = [
        'Absorption_Kinetics_PFO_Qexp(mg/g)', 'Absorption_Kinetics_PFO_Qe cal(mg/g)', 'K1(min-1)',
        'Absorption_Kinetics_PSO_Qe cal(mg/g)', 'Absorption_Kinetics_PSO_K2(mg/g.min)',
        'Isotherm_Langmuir_Qmax(mg/g)', 'Isotherm_Langmuir_KL(L/mg)', 'Isotherm_Freundlich_Kf(mg/g)',
        'Isotherm_Freundlich_1/n', 'PORE VOLUME(cm3/g)', 'SURFACE AREA(m2/g) ',
        'ΔG(kJ /mol)', 'ΔH( kJ/mol)', 'ΔS( J/mol)'
    ]

    # Define feature columns (everything except target columns)
    feature_cols = [col for col in df.columns if col not in target_cols]

    # Handle missing values (if any columns have NaN, you can fill with mean or any other strategy)
    df.fillna(df.mean(), inplace=True)

    # Ensure we are not passing an empty dataset
    if len(df[feature_cols]) == 0 or len(df[target_cols]) == 0:
        st.error("Feature or target columns are empty. Please check your data.")
    else:
        # Split data into train and test
        X = df[feature_cols]
        y = df[target_cols[0]]  # Example: use first target column

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model initialization
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Evaluate model
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        # Display results
        st.write(f'R2 Score: {r2:.4f}')
        st.write(f'RMSE: {rmse:.4f}')
        st.write(f'MAE: {mae:.4f}')
else:
    st.write("Please upload a file to begin.")
