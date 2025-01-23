import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the model and preprocessor
model = joblib.load('linear_regression_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# App title
st.title("Diabetes Risk Prediction App")

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["Introduction", "Prediction"])

# Section: Introduction
if section == "Introduction":
    st.header("Welcome to the Diabetes Risk Prediction App")
    st.markdown("""
    This app predicts the risk of diabetes based on user inputs and provides actionable insights.
    """)

# Section: Prediction
elif section == "Prediction":
    st.header("Predict Diabetes Risk")
    
    # User inputs for prediction
    st.subheader("Enter your details:")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])  # Gender input
    age = st.number_input("Age", min_value=0, max_value=120, value=30)  # Age input
    hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No")  # Hypertension input
    heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No")  # Heart disease input
    smoking_history = st.selectbox("Smoking History", ["never", "current", "former"])  # Smoking history input
    bmi = st.number_input("Body Mass Index", min_value=0.0, max_value=50.0, value=25.0)  # BMI input
    hemoglobin = st.number_input("Hemoglobin A1c Level", min_value=0.0, max_value=20.0, value=5.5)  # Hemoglobin input
    diabetes = st.radio("Previously Diagnosed with Diabetes", [0, 1], format_func=lambda x: "Yes" if x else "No")  # Diabetes input
    
    # Create input data as a DataFrame for preprocessing
    input_data = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "smoking_history": [smoking_history],
        "Body Mass Index": [bmi],
        "Hemoglobin A1c level": [hemoglobin],
        "diabetes": [diabetes]
    })
    
    # Preprocess the input data
    if st.button("Predict"):
        try:
            processed_data = preprocessor.transform(input_data)  # Transform input data
            prediction = model.predict(processed_data)  # Make prediction
            st.write(f"Predicted Blood Glucose Level: {prediction[0]:.2f}")
            
            # Provide risk category based on the prediction
            if prediction[0] > 140:  # Example threshold for "high risk"
                st.error("High Risk of Diabetes! Consult a healthcare professional.")
            elif prediction[0] > 100:
                st.warning("Moderate Risk of Diabetes. Consider lifestyle changes.")
            else:
                st.success("Low Risk of Diabetes. Keep up the healthy habits!")
        except ValueError as e:
            st.error(f"Error: {e}")
