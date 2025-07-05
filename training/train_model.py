import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv('../data/heart_2020_uncleaned.csv')

# Target variable
df['HeartDisease'] = df['HeartDisease'].map({'Yes': 1, 'No': 0})

# Handle missing numeric data
num_cols = ['BMI', 'PhysicalHealth', 'SleepTime']
imputer = SimpleImputer(strategy='median')
df[num_cols] = imputer.fit_transform(df[num_cols])

# Normalize numeric features
scaler = StandardScaler()
numeric_features = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
df[numeric_features] = scaler.fit_transform(df[numeric_features])

# BMI Category (engineered)
def categorize_bmi(bmi_scaled):
    bmi = bmi_scaled * scaler.scale_[0] + scaler.mean_[0]
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'
df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

# Sleep Binning
def bin_sleep(sleep_scaled):
    sleep = sleep_scaled * scaler.scale_[3] + scaler.mean_[3]
    if sleep < 6:
        return 'Short'
    elif sleep <= 8:
        return 'Normal'
    else:
        return 'Long'
df['Sleep_Bin'] = df['SleepTime'].apply(bin_sleep)

# One-hot encoding
df = pd.get_dummies(df, columns=['BMI_Category', 'Sleep_Bin'], drop_first=True)

# Encode all other object columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# Features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Train lighter model
model = RandomForestClassifier(n_estimators=100, max_depth=13, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save model and scaler
joblib.dump(model, '../models/heart_disease_model.pkl', compress=3)
joblib.dump(scaler, '../models/scaler.pkl')

print("✅ Model and Scaler saved in '../models/' folder")
