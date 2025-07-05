import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Setup paths relative to this file
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, "../models/heart_disease_model.pkl"))
scaler = joblib.load(os.path.join(base_dir, "../models/scaler.pkl"))
feature_columns = joblib.load(os.path.join(base_dir, "../models/feature_columns.pkl"))

# App config
st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")
st.title("🩺 Heart Disease Risk Prediction")
st.write("Enter patient information to predict heart disease risk:")

# Collect numeric inputs
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, step=0.1)
physical_health = st.slider("Physical Health (days unwell in last 30 days)", 0, 30)
mental_health = st.slider("Mental Health (days unwell in last 30 days)", 0, 30)
sleep_time = st.slider("Sleep Time (average hours of sleep per day)", 0, 24)

# Binary categorical features
def binary_input(label):
    return st.selectbox(label, ["No", "Yes"]) == "Yes"

smoking = binary_input("Smokes?")
alcohol = binary_input("Drinks Alcohol?")
stroke = binary_input("Had a Stroke?")
diabetic = binary_input("Diabetic?")
diff_walking = binary_input("Difficulty Walking?")
physical_activity = binary_input("Physically Active?")
asthma = binary_input("Has Asthma?")
kidney_disease = binary_input("Has Kidney Disease?")
skin_cancer = binary_input("Has Skin Cancer?")
sex_female = st.selectbox("Sex", ["Male", "Female"]) == "Female"

# BMI category one-hot
def bmi_category_flags(bmi):
    if bmi < 18.5:
        return {'BMI_Category_Normal': 0, 'BMI_Category_Overweight': 0}
    elif bmi < 25:
        return {'BMI_Category_Normal': 1, 'BMI_Category_Overweight': 0}
    elif bmi < 30:
        return {'BMI_Category_Normal': 0, 'BMI_Category_Overweight': 1}
    else:
        return {'BMI_Category_Normal': 0, 'BMI_Category_Overweight': 0}

# Sleep bin one-hot
def sleep_category_flags(sleep):
    if sleep < 6:
        return {'Sleep_Bin_Normal': 0}
    elif sleep <= 8:
        return {'Sleep_Bin_Normal': 1}
    else:
        return {'Sleep_Bin_Normal': 0}

if st.button("Predict Risk"):
    try:
        # Base input dictionary
        input_dict = {
            'BMI': bmi,
            'PhysicalHealth': physical_health,
            'MentalHealth': mental_health,
            'SleepTime': sleep_time,
            'Smoking': int(smoking),
            'AlcoholDrinking': int(alcohol),
            'Stroke': int(stroke),
            'Diabetic': int(diabetic),
            'DiffWalking': int(diff_walking),
            'PhysicalActivity': int(physical_activity),
            'Asthma': int(asthma),
            'KidneyDisease': int(kidney_disease),
            'SkinCancer': int(skin_cancer),
            'Sex_Female': int(sex_female),
        }

        # Add engineered features
        input_dict.update(bmi_category_flags(bmi))
        input_dict.update(sleep_category_flags(sleep_time))

        # Create DataFrame with all expected columns
        input_df = pd.DataFrame([input_dict])

        # Add missing columns as 0
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[feature_columns]

        # Scale the correct numeric columns (same as training)
        numeric_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])

        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🚨 Patient is **At Risk** of Heart Disease.")
        else:
            st.success("✅ Patient is **Not at Risk**.")
        st.write(f"Prediction Confidence: `{probability:.2%}`")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
