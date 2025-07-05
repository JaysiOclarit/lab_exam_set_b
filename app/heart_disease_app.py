# app/heart_disease_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load model and feature names
model = joblib.load(os.path.join('models', 'heart_disease_model.pkl'))
feature_columns = joblib.load(os.path.join('models', 'feature_columns.pkl'))

# Feature mapping
age_categories = [
    '18-24', '25-29', '30-34', '35-39', '40-44', '45-49', 
    '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or older'
]

race_categories = [
    'White', 'Black', 'Asian', 'American Indian/Alaskan Native', 
    'Hispanic', 'Other'
]

health_categories = ['Poor', 'Fair', 'Good', 'Very good', 'Excellent']
diabetic_categories = ['No', 'Yes', 'Borderline', 'Gestational']

# Streamlit app
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="wide")

st.title("Heart Disease Risk Prediction Tool")
st.write("""
This tool predicts the risk of heart disease based on patient health metrics.
Enter the patient information below and click 'Predict' to see the results.
""")

# Create input form
with st.form("patient_info"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographic Information")
        age = st.selectbox("Age Category", age_categories)
        race = st.selectbox("Race", race_categories)
        sex = st.radio("Sex", ["Male", "Female"])

    with col2:
        st.subheader("Health Metrics")
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        physical_health = st.slider("Physical Health Days (unwell past 30 days)", 0, 30, 0)
        mental_health = st.slider("Mental Health Days (unwell past 30 days)", 0, 30, 0)
        sleep_time = st.slider("Sleep Hours (per day)", 1, 24, 7)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Health Conditions")
        smoking = st.radio("Smoking", ["No", "Yes"])
        alcohol = st.radio("Alcohol Drinking", ["No", "Yes"])
        stroke = st.radio("History of Stroke", ["No", "Yes"])
        diff_walking = st.radio("Difficulty Walking", ["No", "Yes"])

    with col4:
        st.subheader("")
        asthma = st.radio("Asthma", ["No", "Yes"])
        kidney_disease = st.radio("Kidney Disease", ["No", "Yes"])
        skin_cancer = st.radio("Skin Cancer", ["No", "Yes"])
        diabetic = st.selectbox("Diabetic", diabetic_categories)

    physical_activity = st.radio("Physical Activity", ["No", "Yes"])
    gen_health = st.select_slider("General Health", health_categories)

    submit_button = st.form_submit_button("Predict")

# Process input and make prediction
if submit_button:
    input_data = {
        'BMI': [bmi],
        'Smoking': [1 if smoking == "Yes" else 0],
        'AlcoholDrinking': [1 if alcohol == "Yes" else 0],
        'Stroke': [1 if stroke == "Yes" else 0],
        'PhysicalHealth': [physical_health],
        'MentalHealth': [mental_health],
        'DiffWalking': [1 if diff_walking == "Yes" else 0],
        'Sex': [sex],
        'AgeCategory': [age],
        'Race': [race],
        'Diabetic': [diabetic],
        'PhysicalActivity': [1 if physical_activity == "Yes" else 0],
        'GenHealth': [gen_health],
        'SleepTime': [sleep_time],
        'Asthma': [1 if asthma == "Yes" else 0],
        'KidneyDisease': [1 if kidney_disease == "Yes" else 0],
        'SkinCancer': [1 if skin_cancer == "Yes" else 0]
    }

    input_df = pd.DataFrame(input_data)

    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"**At Risk** (Probability: {probability:.2%})")
            st.write("""
            The patient is at risk of heart disease. Consider:
            - Comprehensive cardiac evaluation
            - Lifestyle modification counseling
            - Risk factor management
            """)
        else:
            st.success(f"**Not At Risk** (Probability: {1 - probability:.2%})")
            st.write("""
            The patient shows no significant heart disease risk. 
            Maintain healthy habits:
            - Regular physical activity
            - Balanced nutrition
            - Annual health checkups
            """)

        st.progress(probability)
        st.caption(f"Risk Confidence: {probability:.2%}")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

st.sidebar.header("About This Tool")
st.sidebar.info("""
This clinical decision support tool predicts heart disease risk using 
machine learning models trained on CDC behavioral risk data.

**Key Features:**
- Evaluates 17 health and demographic factors
- Provides probability-based risk assessment
- Identifies high-risk patients for early intervention

**Clinical Use:**
- Screening tool for primary care
- Patient education resource
- Risk stratification for preventive care

*Always use clinical judgment alongside predictions.*
""")
