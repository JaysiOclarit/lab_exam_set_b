# src/utils.py
import os
import joblib

class Utils:
    @staticmethod
    def save_model(model, X_columns):
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, os.path.join('models', 'heart_disease_model.pkl'))
        joblib.dump(X_columns, os.path.join('models', 'feature_columns.pkl'))
        print("Model and features saved successfully!")
