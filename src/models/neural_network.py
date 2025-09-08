import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.sparse import issparse
from .nn_text_classifier import NNTextClassifier
from scipy.sparse import issparse


class NeuralNetworkModel(NNTextClassifier):
    """Neural network classifier for AI text detection."""

    def __init__(self, encoder):
        """Initialize with encoder and parent class."""
        
        # Get the device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self._device} device") 

        # Initialize the parent class with the encoder
        super().__init__(encoder)

        # Define function to flatten the input
        self._flatten = nn.Flatten()

        # Layers will be initialized after we know the input size
        self._layers = None

    def forward(self, x):
        """Run the forward pass."""
        x = self._flatten(x)
        
        if self._layers is None:
            raise RuntimeError("Model layers have not been initialized. Call train_model first.")
            
        logits = self._layers(x)
        return logits

    def train_model(self, X, y, epochs=5, lr=0.001):
        """Train the model."""

        X_iter = tqdm(X, desc="Encoding training data", leave=True)

        # Transform the data
        X_vectors = self._encoder.transform(X_iter)

        # Convert to dense array if sparse
        if issparse(X_vectors):
            X_vectors = X_vectors.toarray()
            
        if self._layers is None:
            input_size = X_vectors.shape[1]
            self._layers = nn.Sequential(
                nn.Linear(input_size, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 2),
            ).to(self._device)
            
        # Convert the data to tensors
        X_tensor = torch.tensor(X_vectors, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        # Define the dataset & dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Define the loss function & optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        # Train the model
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self._device), batch_y.to(self._device)

                optimizer.zero_grad()
                logits = self.forward(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    def predict(self, text):
        """Make predictions on the text."""
        self.eval()
        with torch.no_grad():
            # Get encoded input
            X_vector = self._encoder.transform([text] if isinstance(text, str) else text)
            
            # Convert to dense array if sparse
            if issparse(X_vector):
                X_vector = X_vector.toarray()
                
            # Convert to tensor and move to device
            X_tensor = torch.tensor(X_vector, dtype=torch.float32).to(self._device)

            # Get predictions
            logits = self.forward(X_tensor)
            predictions = logits.argmax(dim=1).cpu().numpy()
            probabilities = F.softmax(logits, dim=1).cpu().numpy()
            
        return predictions, probabilities

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        self.eval()
        with torch.no_grad():
            # Get predictions
            predictions, _ = self.predict(X_test)
            
            # Calculate metrics
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