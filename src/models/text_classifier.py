from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

class TextClassifier(ABC):
    """Base class for text classification models."""
    
    def __init__(self, vectorizer=None):
        """Initialize with optional vectorizer."""
        self._vectorizer = vectorizer or TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 3),
            max_features=5000,
            min_df=2,
            max_df=0.95
        )
    
    @abstractmethod
    def train(self, X, y):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, text):
        """Make predictions on text."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        pass
    
    def _vectorize_text(self, text):
        """Transform text using the vectorizer."""

        # If the text is a string, convert it to a list
        if isinstance(text, str):
            text = [text]

        # Vectorize the text
        return self._vectorizer.transform(text)
    
    def save(self, folder: str, filename: str):
        """Save model to file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Create the path
        path = os.path.join(folder, filename)

        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load model from file."""
        
        with open(path, 'rb') as f:
            return pickle.load(f)