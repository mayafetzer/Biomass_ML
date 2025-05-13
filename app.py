import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import MinMaxScaler

# Streamlit UI
st.title("Pharmaceutical Data Analysis and ML Modeling")

# Step 1: File upload
uploaded_file = st.file_uploader("Upload your Excel dataset", type="xlsx")

if uploaded_file is not None:
    # Read the uploaded Excel file into a DataFrame
    df = pd.read_excel(uploaded_file)

    # Display the first few rows of the dataframe
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Step 2: Option to drop or keep categorical variables
    drop_categorical = st.checkbox("Drop Categorical Variables", value=False)

    # Step 3: Data Preprocessing
    st.subheader("Data Preprocessing")
    target_columns = [
        'Absorption_Kinetics_PFO_Qexp(mg/g)', 'Absorption_Kinetics_PFO_Qe cal(mg/g)', 'K1(min-1)',
        'Absorption_Kinetics_PSO_Qe cal(mg/g)', 'Absorption_Kinetics_PSO_K2(mg/g.min)',
        'Isotherm_Langmuir_Qmax(mg/g)', 'Isotherm_Langmuir_KL(L/mg)', 'Isotherm_Freundlich_Kf(mg/g)',
        'Isotherm_Freundlich_1/n', 'PORE VOLUME(cm3/g)', 'SURFACE AREA(m2/g) ',
        'ΔG(kJ /mol)', 'ΔH( kJ/mol)', 'ΔS( J/mol)'
    ]

    # Clean columns and handle missing values
    for col in target_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: np.nan if isinstance(x, str) and x.strip() == '-' else x)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)

    # Step 4: Handle Categorical Variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    if drop_categorical:
        df = df.drop(columns=categorical_cols)

    # Normalize the data
    feature_cols = [col for col in df.columns if col not in target_columns]
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[feature_cols + target_columns]), columns=feature_cols + target_columns)

    # Step 5: Train and Evaluate Machine Learning Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'SVR': SVR(),
        'KNN': KNeighborsRegressor()
    }

    results = []
    for target in target_columns:
        X = df[feature_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                'Target': target,
                'Model': model_name,
                'R2': r2_score(y_test, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                'MAE': mean_absolute_error(y_test, y_pred)
            })

    results_df = pd.DataFrame(results)

    # Display the results
    st.subheader("Model Evaluation Results")
    metrics = ['R2', 'RMSE', 'MAE']
    for metric in metrics:
        st.write(f"### {metric} Score by Model per Target")
        fig = plt.figure(figsize=(16, 6))
        sns.barplot(data=results_df, x='Target', y=metric, hue='Model')
        plt.title(f'{metric} Score by Model per Target')
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(fig)

    # Show Summary Table
    summary_table = results_df[['Model', 'R2', 'RMSE', 'MAE']].groupby('Model').mean().reset_index()
    st.subheader("Summary Table")
    st.write(summary_table)

    # Step 6: Partial Dependence Plots
    best_model_name = summary_table.loc[summary_table['R2'].idxmax(), 'Model']
    st.write(f"Best Model: {best_model_name}")

    if best_model_name in models:
        best_model = models[best_model_name]
        X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df[target_columns[0]], test_size=0.2, random_state=42)
        best_model.fit(X_train, y_train)

        features_to_plot = feature_cols[:5]  # Adjust this list as needed
        fig, ax = plt.subplots(figsize=(12, 8))
        PartialDependenceDisplay.from_estimator(best_model, X_train, features_to_plot, ax=ax)
        plt.suptitle(f"Partial Dependence Plots for {best_model_name}")
        plt.tight_layout()
        st.pyplot(fig)

# Save model as pickle file
if st.button("Save Best Model"):
    best_model = models[best_model_name]
    with open('pharma_categorical.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    st.success("Best model saved as 'pharma_categorical.pkl'")
