from abc import ABC, abstractmethod
import os
import pickle

class TextEncoder(ABC):
    """Base class for text encoders."""

    @abstractmethod
    def fit(self, texts, y=None):
        """Fit the encoder to the texts (if applicable)."""
        pass

    @abstractmethod
    def transform(self, texts):
        """Transform the texts."""
        pass

    def fit_transform(self, texts, y=None):
        """Fit the encoder and transform the texts."""
        self.fit(texts, y)
        return self.transform(texts)

    def save(self, folder: str, filename: str):
        """Save the encoder to a file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Build the path
        path = os.path.join(folder, filename)

        # Save the model
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, folder: str, filename: str):
        """Load the encoder from a file."""

        # Create the folder if it doesn't exist
        os.makedirs(folder, exist_ok=True)

        # Build the path
        path = os.path.join(folder, filename)
        
        # Load the model
        with open(path, 'rb') as f:
            return pickle.load(f)
