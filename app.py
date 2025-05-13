import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer # Import SimpleImputer
import pickle # Import pickle
from io import BytesIO  # Import BytesIO for handling file-like objects

# Load data function (modified to take file upload)
def load_data(uploaded_file, model_type, use_categorical):
    """
    Loads and preprocesses the dataset from a file uploaded by the user.

    Args:
        uploaded_file: The file uploaded by the user (CSV or Excel).
        model_type (str):  'pharma' or 'dye'.
        use_categorical (bool): Whether to include categorical variables.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, df) if successful, (None, None, None, None, None) otherwise.
    """
    if uploaded_file is not None:
        try:
            # Use BytesIO to handle the file-like object
            bytes_data = uploaded_file.read()
            try:
                # Try reading as CSV
                df = pd.read_csv(BytesIO(bytes_data))
            except Exception as e:
                try:
                    # If CSV fails, try reading as Excel
                    df = pd.read_excel(BytesIO(bytes_data))
                except Exception as e:
                    st.error(f"Error: Could not read the file.  Please upload a valid CSV or Excel file.  Error Details: {e}")
                    return None, None, None, None, None

        # Common preprocessing steps
        try: # Added try and except
            df = df.dropna()
            df = df.drop_duplicates()
        except Exception as e:
            st.error(f"Error during data preprocessing: {e}")
            return None, None, None, None, None

        # Model-specific preprocessing
        if model_type == 'pharma':
            if use_categorical:
                X = df[['MW', 'logP', 'HBA', 'HBD', 'RotB', 'Amide', 'AromaticRings', 'ReducibleBonds', 'QED', 'TPSA', 'FCSP3', 'caco2', 'Pappab', 'pgp', 'HLM_CLint', 'mouse_CLint', 'BBB', 'IVIVC_rat', 'F', 'fu']]
                y = df['Yield']
                X = pd.get_dummies(X, drop_first=True)
            else:
                X = df[['MW', 'logP', 'HBA', 'HBD', 'RotB', 'Amide', 'AromaticRings', 'ReducibleBonds', 'QED', 'TPSA', 'FCSP3']]
                y = df['Yield']
        elif model_type == 'dye':
            if use_categorical:
                X = df[['peak_absorbance', 'Molar_Extinction_Coefficient', 'Dye_class']]
                y = df['Brightness']
                X = pd.get_dummies(X, drop_first=True)
            else:
                X = df[['peak_absorbance', 'Molar_Extinction_Coefficient']]
                y = df['Brightness']
        else:
            st.error("Invalid model type. Please choose 'pharma' or 'dye'.")
            return None, None, None, None, None

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test, df
    return None, None, None, None, None


# Model training function
def train_model(X_train, y_train):
    """
    Trains a RandomForestRegressor model.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        RandomForestRegressor: Trained model.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Prediction function
def predict_and_evaluate(model, X_test, y_test):
    """
    Makes predictions and evaluates the model.

    Args:
        model (RandomForestRegressor): Trained model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.

    Returns:
        tuple: (predictions, mse, r2)
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return predictions, mse, r2

# Main Streamlit app
def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("Biomass ML App")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel file)", type=["csv", "xlsx"])

    # Model type selection
    model_type = st.selectbox("Select Model Type", ['pharma', 'dye'])

    # Include categorical variables
    use_categorical = st.checkbox("Include Categorical Variables")

    # Load data and train model on button click
    if st.button("Load Data and Train Model"):
        X_train, X_test, y_train, y_test, df = load_data(uploaded_file, model_type, use_categorical)
        if X_train is not None: # Check if data loading was successful
            model = train_model(X_train, y_train)
            predictions, mse, r2 = predict_and_evaluate(model, X_test, y_test)

            # Display results
            st.subheader("Model Evaluation")
            st.write(f"Mean Squared Error: {mse:.2f}")
            st.write(f"R-squared: {r2:.2f}")

            st.subheader("Predictions vs Actual Values")
            results_df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
            st.dataframe(results_df)
        else:
            st.write("Please upload a valid dataset to proceed.") # added message

if __name__ == "__main__":
    main()
