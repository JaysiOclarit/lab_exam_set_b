# main.py
from src.data_loader import DataLoader
from src.data_preprocessor import DataPreprocessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.utils import Utils
import os
import pandas as pd

file_path = os.path.join('data', 'heart_2020_uncleaned.csv')
loader = DataLoader(file_path)
df = loader.load_data()

preprocessor = DataPreprocessor()
X, y = preprocessor.clean_and_prepare(df)

trainer = ModelTrainer(X, y)
X_train, X_test, y_train, y_test = trainer.get_data_splits()
models = trainer.get_models()

results = {}
best_model = None
best_score = 0

for name, model in models.items():
    print(f"\nTraining: {name}")
    pipeline = preprocessor.get_pipeline(model)
    pipeline.fit(X_train, y_train)
    metrics = Evaluator.evaluate_model(pipeline, X_test, y_test)
    results[name] = metrics

    if metrics['ROC AUC'] > best_score:
        best_model = pipeline
        best_score = metrics['ROC AUC']
        best_name = name

print(f"\nBest model: {best_name} with ROC AUC: {best_score:.4f}")
Utils.save_model(best_model, list(X.columns))

results_df = pd.DataFrame(results).T
print("\nModel Comparison:")
print(results_df)