import streamlit as st
import pickle
import os
import pandas as pd

def load_model(file_path):
    """Loads a pickled model from the given file path."""
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

def predict(model, features):
    """Makes a prediction using the loaded model and input features."""
    try:
        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)
        return prediction[0]
    except Exception as e:
        return f"Error during prediction: {e}"

def main():
    st.title("Biomass Prediction App")

    # Find all pickle files in the current directory
    pickle_files = [f for f in os.listdir() if f.endswith(".pkl")]

    if not pickle_files:
        st.warning("No pickle files found in the current directory.")
        return

    # Allow user to select a pickle file
    selected_file = st.selectbox("Select a trained model:", pickle_files)

    # Load the selected model
    model = load_model(selected_file)
    st.success(f"Model '{selected_file}' loaded successfully!")

    # Get feature names from the model (if possible)
    feature_names = None
    if hasattr(model, 'feature_names_in_'):
        feature_names = list(model.feature_names_in_)
    elif hasattr(model, 'n_features_in_'):
        feature_names = [f"feature_{i+1}" for i in range(model.n_features_in_)]

    if feature_names:
        st.subheader("Enter Feature Values:")
        features = {}
        for feature in feature_names:
            value = st.number_input(f"{feature}:")
            features[feature] = value

        if st.button("Predict Biomass"):
            if features:
                prediction = predict(model, features)
                st.subheader("Prediction:")
                st.write(f"The predicted biomass is: {prediction}")
            else:
                st.warning("Please enter all feature values.")
    else:
        st.warning("Could not automatically determine feature names. Please refer to the model documentation for the required input features.")
        # Provide generic input fields if feature names cannot be determined
        num_features = st.number_input("Enter the number of features expected by the model:", min_value=1, step=1)
        features = {}
        for i in range(num_features):
            value = st.number_input(f"Feature {i+1}:")
            features[f"feature_{i+1}"] = value

        if st.button("Predict Biomass"):
            if features and len(features) == num_features:
                prediction = predict(model, list(features.values())) # Pass values in order
                st.subheader("Prediction:")
                st.write(f"The predicted biomass is: {prediction}")
            else:
                st.warning(f"Please enter all {num_features} feature values.")

if __name__ == "__main__":
    main()
