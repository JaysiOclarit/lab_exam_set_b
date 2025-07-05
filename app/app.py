import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

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
sex = st.selectbox("Sex", ["Male", "Female"]) == "Female"

# BMI category (one-hot encoded: Normal, Overweight; Obese is the base case)
def bmi_category_flags(bmi):
    if bmi < 18.5:
        return [1, 0, 0]  # Underweight
    elif bmi < 25:
        return [0, 1, 0]  # Normal
    elif bmi < 30:
        return [0, 0, 1]  # Overweight
    else:
        return [0, 0, 0]  # Obese (drop_first=True)

# Sleep bin (one-hot encoded: Short, Normal; Long is the base case)
def sleep_category_flags(sleep):
    if sleep < 6:
        return [1, 0]  # Short
    elif sleep <= 8:
        return [0, 1]  # Normal
    else:
        return [0, 0]  # Long (drop_first=True)

if st.button("Predict Risk"):
    try:
        # Step 1: Numeric features scaled
        numeric_data = np.array([[bmi, physical_health, mental_health, sleep_time]])
        numeric_scaled = scaler.transform(numeric_data).flatten()

        # Step 2: Binary encoded features
        binary_data = [
            int(smoking), int(alcohol), int(stroke), int(diabetic),
            int(diff_walking), int(physical_activity), int(asthma),
            int(kidney_disease), int(skin_cancer), int(sex)
        ]

        # Step 3: Engineered features (BMI category & sleep bin)
        bmi_flags = bmi_category_flags(bmi)[1:]  # drop_first=True → skip Underweight
        sleep_flags = sleep_category_flags(sleep_time)[1:]  # drop_first=True → skip Short

        # Step 4: Combine all features
        final_input = np.concatenate([numeric_scaled, binary_data, bmi_flags, sleep_flags]).reshape(1, -1)

        # Step 5: Predict
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]

        # Step 6: Display result
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("🚨 Patient is **At Risk** of Heart Disease.")
        else:
            st.success("✅ Patient is **Not at Risk**.")

        st.write(f"Prediction Confidence: `{probability:.2%}`")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
