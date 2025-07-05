# src/model_trainer.py
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

    def get_data_splits(self):
        return self.X_train, self.X_test, self.y_train, self.y_test

    def get_models(self):
        return self.models