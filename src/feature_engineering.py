# src/feature_engineering.py
import pandas as pd

class FeatureEngineer:
    @staticmethod
    def add_bmi_category(df):
        def bmi_category(bmi):
            if bmi < 18.5:
                return 'Underweight'
            elif bmi < 25:
                return 'Normal'
            elif bmi < 30:
                return 'Overweight'
            else:
                return 'Obese'

        df['BMI_Category'] = df['BMI'].apply(bmi_category)
        return df

    @staticmethod
    def add_interactions(df):
        df['BMI_PhysicalHealth'] = df['BMI'] * df['PhysicalHealth']
        df['BMI_MentalHealth'] = df['BMI'] * df['MentalHealth']
        df['BMI_SleepTime'] = df['BMI'] * df['SleepTime']
        df['BMI_DiffWalking'] = df['BMI'] * df['DiffWalking']
        df['BMI_PhysicalActivity'] = df['BMI'] * df['PhysicalActivity'] if 'PhysicalActivity' in df.columns else df['BMI'] * 0
        df['BMI_Asthma'] = df['BMI'] * df['Asthma'] if 'Asthma' in df.columns else df['BMI'] * 0
        df['BMI_KidneyDisease'] = df['BMI'] * df['KidneyDisease'] if 'KidneyDisease' in df.columns else df['BMI'] * 0
        df['BMI_SkinCancer'] = df['BMI'] * df['SkinCancer'] if 'SkinCancer' in df.columns else df['BMI'] * 0
        return df

    @staticmethod
    def apply(df):
        df = FeatureEngineer.add_bmi_category(df)
        df = FeatureEngineer.add_interactions(df)
        return df
