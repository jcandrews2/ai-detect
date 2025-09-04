from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn

class NNTextClassifier(nn.Module, ABC):
    """Base class for neural network text classification models."""
    
    def __init__(self, encoder):
        """Initialize with encoder and parent class."""
        self._encoder = encoder

        super().__init__()

    @abstractmethod
    def forward(self, x):
        """Run the forward pass."""
        pass

    @abstractmethod
    def train_model(self, X, y, epochs=5, lr=0.001):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, text):
        """Make predictions on the text."""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate the model."""
        pass

    def save(self, folder: str, filename: str):
        """Save the model to a file."""
        os.makedirs(folder, exist_ok=True)
        torch.save(self, os.path.join(folder, filename))

    @classmethod
    def load(cls, folder: str, filename: str):
        """Load the model from a file."""
        model = torch.load(os.path.join(folder, filename))
        model.eval()
        return model
