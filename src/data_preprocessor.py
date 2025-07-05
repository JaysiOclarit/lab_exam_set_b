# src/data_preprocessor.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.preprocessor = None

    def clean_and_prepare(self, df):
        df.columns = df.columns.str.strip()
        binary_cols = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 
                       'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']
        df[binary_cols] = df[binary_cols].apply(lambda x: x.map({'Yes': 1, 'No': 0}))
        df['Diabetic'] = df['Diabetic'].replace({
            'No, borderline diabetes': 'Borderline', 
            'Yes (during pregnancy)': 'Gestational'
        })
        df['SleepTime'] = pd.to_numeric(df['SleepTime'], errors='coerce')

        X = df.drop('HeartDisease', axis=1)
        y = df['HeartDisease'].map({'Yes': 1, 'No': 0})

        numerical = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
        categorical = [col for col in X.columns if col not in numerical]

        self.preprocessor = ColumnTransformer(transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())]), numerical),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical)
        ])
        return X, y

    def get_pipeline(self, model):
        return Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', model)
        ])