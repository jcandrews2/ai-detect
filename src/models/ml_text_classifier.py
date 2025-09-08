from abc import ABC, abstractmethod
import pickle
import os

class MLTextClassifier(ABC):
    """Base class for machine learning text classification models."""
    
    def __init__(self, encoder):
        """Initialize with encoder."""
        self._encoder = encoder
    
    @abstractmethod
    def train_model(self, X, y):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, text):
        """Make predictions on the text."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        pass

    
    def save(self, folder: str, filename: str):
        """Save the model to a file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Build the path
        path = os.path.join(folder, filename)

        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, folder: str, filename: str):
        """Load the model from a file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Build the path
        path = os.path.join(folder, filename)
        
        # Load the model
        with open(path, 'rb') as f:
            return pickle.load(f)