# src/data_loader.py
import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        print(f"Dataset shape: {df.shape}")
        print("\nData types:\n", df.dtypes)
        print("\nMissing values:\n", df.isnull().sum())
        print("\nTarget distribution:\n", df['HeartDisease'].value_counts(normalize=True))
        return df