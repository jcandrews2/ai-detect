from abc import ABC, abstractmethod
import pickle
import os

class TextClassifier(ABC):
    """Base class for text classification models."""
    
    def __init__(self, vectorizer=None):
        """Initialize with optional vectorizer."""
        self._vectorizer = vectorizer
    
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
    
    def save(self, folder: str, filename: str):
        """Save model to file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Build the path
        path = os.path.join(folder, filename)

        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, folder: str, filename: str):
        """Load model from file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Build the path
        path = os.path.join(folder, filename)
        
        # Load the model
        with open(path, 'rb') as f:
            return pickle.load(f)