from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class RandomForest:
    """A random forest classifier for text classification."""
    
    def __init__(self, data_loader):
        """Initialize with a DataLoader instance."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=0)
        self.data_loader = data_loader

    def train(self):
        """Train the model using data from the DataLoader."""
        X_train, y_train = self.data_loader.get_train_data()
        print("Training the model...")
        self.model.fit(X_train, y_train)
        return self

    def predict(self, text):
        """Predict the class of a text snippet."""
        X = self.data_loader.vectorize_text(text)
        return self.model.predict(X), self.model.predict_proba(X)

    def evaluate(self):
        """Evaluate the model with precision, recall, f1, and accuracy."""
        X_test, y_test = self.data_loader.get_test_data()
        predictions = self.model.predict(X_test)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)

        print(f"\nEvaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}\n")
        
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy
        }