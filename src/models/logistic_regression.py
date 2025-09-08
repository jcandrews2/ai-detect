from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .ml_text_classifier import MLTextClassifier
from tqdm import tqdm

class LogisticRegressionModel(MLTextClassifier):
    """Logistic regression classifier for AI text detection."""
    
    def __init__(self, encoder):
        """Initialize the logistic regression classifier."""

        # Initialize the parent class with the encoder
        super().__init__(encoder)

        # Initialize the classifier
        self._classifier = LogisticRegression(
            max_iter=1000,
            random_state=0,
            verbose=2,
            C=1.0,
            class_weight='balanced',
            n_jobs=-1
        )

    def train_model(self, X_train, y_train):
        """Train the model on provided data."""
        
        X_iter = tqdm(X_train, desc="Encoding training data", leave=True)

        # Transform the data
        X_vectors = self._encoder.transform(X_iter)

        # Train the classifier
        self._classifier.fit(X_vectors, y_train)

    def predict(self, text):
        """Predict whether text is AI-generated."""

        # Transform the text
        X_vector = self._encoder.transform([text] if isinstance(text, str) else text)

        # Predict the class and probability
        return (
            self._classifier.predict(X_vector),
            self._classifier.predict_proba(X_vector)
        )

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""

        # Transform the data
        X_vectors = self._encoder.transform(X_test)

        # Predict the class
        predictions = self._classifier.predict(X_vectors)

        # Calculate the metrics
        metrics = {
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1': f1_score(y_test, predictions),
            'accuracy': accuracy_score(y_test, predictions)
        }

        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.title()}: {value:.4f}")
        print()
    