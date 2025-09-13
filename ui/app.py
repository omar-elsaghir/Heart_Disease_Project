import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- 1. LOAD THE TRAINED MODEL AND SCALER ---
# Load the model and scaler from the files you saved
try:
    model = joblib.load('models/final_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Make sure 'final_model.pkl' and 'scaler.pkl' are in a 'models' folder.")
    st.stop()

# --- 2. DEFINE THE USER INTERFACE ---
st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# Title for the app
st.title("❤️ Heart Disease Prediction App")
st.write(
    "This app predicts the likelihood of a patient having heart disease based on their medical data. Please enter the patient's information in the sidebar.")

# Sidebar for user inputs
st.sidebar.header("Patient Medical Data")


# Function to collect user inputs
def user_input_features():
    # Create sliders and select boxes for all the features your model was trained on
    age = st.sidebar.slider('Age', 20, 80, 50)
    sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
    cp = st.sidebar.selectbox('Chest Pain Type', (1, 2, 3, 4), format_func=lambda x:
    {1: 'Typical Angina', 2: 'Atypical Angina', 3: 'Non-Anginal Pain', 4: 'Asymptomatic'}[x])
    trestbps = st.sidebar.slider('Resting Blood Pressure (mm Hg)', 90, 200, 120)
    thalach = st.sidebar.slider('Maximum Heart Rate Achieved', 70, 220, 150)
    exang = st.sidebar.selectbox('Exercise Induced Angina', ('No', 'Yes'))
    oldpeak = st.sidebar.slider('ST Depression Induced by Exercise', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', (1, 2, 3),
                                 format_func=lambda x: {1: 'Upsloping', 2: 'Flat', 3: 'Downsloping'}[x])
    ca = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy', (0, 1, 2, 3))
    thal = st.sidebar.selectbox('Thalassemia', (3, 6, 7),
                                format_func=lambda x: {3: 'Normal', 6: 'Fixed Defect', 7: 'Reversible Defect'}[x])

    # Convert categorical inputs to numerical format the model expects
    sex_num = 1 if sex == 'Male' else 0
    exang_num = 1 if exang == 'Yes' else 0

    # Create a dictionary of the inputs
    data = {
        'age': age,
        'sex': sex_num,
        'cp': cp,
        'trestbps': trestbps,
        'thalach': thalach,
        'exang': exang_num,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }

    # Convert dictionary to a pandas DataFrame
    features = pd.DataFrame(data, index=[0])
    return features


# Get user input
input_df = user_input_features()

# Display the user's input
st.subheader("Patient's Input Data")
st.write(input_df)

# --- 3. PROCESS INPUT AND MAKE PREDICTION ---
if st.button('Predict'):
    # Ensure the order of columns in the input_df matches the model's training order
    # This list must be the same as 'selected_features' from your notebook
    model_features = ['age', 'sex', 'cp', 'trestbps', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = input_df[model_features]

    # Scale the numerical features using the loaded scaler
    # This list must be the same as 'numerical_cols' from your notebook
    numerical_cols = ['age', 'trestbps', 'thalach', 'oldpeak', 'ca']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Make prediction
    prediction_proba = model.predict_proba(input_df)

    # Get the probability of having heart disease (the second class)
    probability_of_disease = prediction_proba[0][1]

    st.subheader('Prediction Result')
    st.write(f"**Probability of having Heart Disease: {probability_of_disease:.2f}**")

    # Display the result with a nice visual
    if probability_of_disease > 0.5:
        st.error("The model predicts a HIGH risk of heart disease.")
    else:
        st.success("The model predicts a LOW risk of heart disease.")