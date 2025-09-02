from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from .text_classifier import TextClassifier
from tqdm import tqdm

class LogisticRegression(TextClassifier):
    """Random forest classifier for AI text detection."""
    
    def __init__(self, vectorizer=None):
        """Initialize the random forest classifier."""

        # Initialize the parent class with the vectorizer
        super().__init__(vectorizer)

        # Initialize the classifier
        self._classifier = LogisticRegression(
            max_iter=100,
            random_state=0,
            verbose=2
        )

    def _vectorize_text(self, text):
        """Transform text using the vectorizer."""

        # If the text is a string, convert it to a list
        if isinstance(text, str):
            text = [text]

        text_iter = tqdm(text, desc="Vectorizing test data", leave=True)

        # Vectorize the text
        return self._vectorizer.transform(text_iter)

    def train(self, X, y):
        """Train the model on provided data."""
        
        X_iter = tqdm(X, desc="Vectorizing training data", leave=True)

        # Fit vectorizer if not already fitted
        if not hasattr(self._vectorizer, 'vocabulary_'):
            self._vectorizer.fit(X_iter)
        
        # Transform text and train
        X_vectors = self._vectorizer.transform(X_iter)

        # Train the classifier
        self._classifier.fit(X_vectors, y)

    def predict(self, text):
        """Predict whether text is AI-generated."""

        # Vectorize the text
        X_vector = self._vectorize_text(text)

        # Predict the class and probability
        return (
            self._classifier.predict(X_vector),
            self._classifier.predict_proba(X_vector)
        )

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""

        # Vectorize the test data
        X_vectors = self._vectorize_text(X_test)

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
    