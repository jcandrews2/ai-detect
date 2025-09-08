from typing import Literal
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.models.ml_text_classifier import MLTextClassifier
from src.models.nn_text_classifier import NNTextClassifier
from src.data.data_loader import DataLoader
from src.encoders.tfidf_encoder import TfidfEncoder
from src.encoders.glove_encoder import GloveEncoder
from src.encoders.text_encoder import TextEncoder
import uvicorn
from src.api.api import app
import os

def select_encoder(
    encoder_type: Literal["tfidf", "glove"],
) -> tuple[TextEncoder, bool]:
    """Select the encoder."""

    encoder_classes = {
        "tfidf": [TfidfEncoder, True],
        "glove": [GloveEncoder, False]
    }

    return encoder_classes[encoder_type]

def select_model(
    model_type: Literal["random_forest", "logistic_regression", "neural_network"]
) -> MLTextClassifier | NNTextClassifier:
    """Select the model."""
    model_classes = {
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "neural_network": NeuralNetworkModel
    }
    return model_classes[model_type]


def train_and_evaluate_model(
    model_type: Literal["random_forest", "logistic_regression", "neural_network"],
    encoder_type: Literal["tfidf", "glove"],
    data_path: str,
    model_save_path: str,
    model_filename: str,
    encoder_save_path: str,
    encoder_filename: str
) -> None:
    """Train and evaluate a model for AI text detection

    Args:
        model_type: Type of model to use ("random_forest" or "logistic_regression" or "neural_network")
        encoder_type: Type of encoder to use ("tfidf" or "glove")
        data_path: Path to the data CSV file
        model_save_path: Directory to save the trained model
        model_filename: Filename for the saved model
        encoder_save_path: Directory to save the trained encoder
        encoder_filename: Filename for the saved encoder

    Returns:
        None
    """

    print(f"\n=== {model_type} + {encoder_type} ===")

    # Load and prepare the data
    print("\n[ 1/4 ] Preparing data...")
    data_loader = DataLoader(data_path)
    data_loader.prepare()
    
    X_train, y_train = data_loader.get_train_data()
    X_test, y_test = data_loader.get_test_data()

    # Select the encoder
    encoder_class, should_fit_encoder = select_encoder(encoder_type)

    # Load or train the encoder
    print("\n[ 2/4 ] Loading encoder...")
    encoder_path = os.path.join(encoder_save_path, encoder_filename)
    if os.path.exists(encoder_path):
        encoder = encoder_class.load(encoder_save_path, encoder_filename)
    else:
        # Initialize the encoder
        encoder = encoder_class()

        # Fit the encoder if needed
        if should_fit_encoder:
            encoder.fit(X_train)

        # Save the encoder
        encoder.save(encoder_save_path, encoder_filename)

    # Select the model
    model_class = select_model(model_type)

    # Load or train the model
    print("\n[ 3/4 ] Loading model...")
    model_path = os.path.join(model_save_path, model_filename)
    if os.path.exists(model_path):
        model = model_class.load(model_save_path, model_filename)
    else:
        
        # Initialize the model
        model = model_class(encoder)

        # Train the model
        model.train_model(X_train, y_train)

        # Save the model
        model.save(model_save_path, model_filename)

    print("\n[ 4/4 ] Evaluating model...")
    model.evaluate(X_test, y_test)


def main():
    # dataset = "~/Downloads/AI_Human.csv"
    dataset = "~/Downloads/balanced_ai_human_prompts.csv"

    """
    Args:
        model_type: Type of model to use ("neural_network")
        encoder_type: Type of encoder to use ("tfidf" or "glove")
        data_path: Path to the data CSV file
        model_save_path: Directory to save the trained model
        model_filename: Filename for the saved model
        encoder_save_path: Directory to save the trained encoder
        encoder_filename: Filename for the saved encoder

    """

    # Random Forest model with GloVe encoder
    train_and_evaluate_model(
        model_type="random_forest",
        encoder_type="glove",
        data_path=dataset,
        model_save_path="src/models/saved",
        model_filename="rf_model_glove.pkl",
        encoder_save_path="src/encoders/saved",
        encoder_filename="glove.pkl"
    )

    # Random Forest model with TF-IDF encoder
    train_and_evaluate_model(
        model_type="random_forest",
        encoder_type="tfidf",
        data_path=dataset,
        model_save_path="src/models/saved",
        model_filename="rf_model_tfidf.pkl",
        encoder_save_path="src/encoders/saved",
        encoder_filename="tfidf.pkl"
    )

    # Logistic Regression model with GloVe encoder
    train_and_evaluate_model(
        model_type="logistic_regression",
        encoder_type="glove",
        data_path=dataset,
        model_save_path="src/models/saved",
        model_filename="lr_model_glove.pkl",
        encoder_save_path="src/encoders/saved",
        encoder_filename="glove.pkl"
    )

    # Logistic Regression model with TF-IDF encoder
    train_and_evaluate_model(
        model_type="logistic_regression",
        encoder_type="tfidf",
        data_path=dataset,
        model_save_path="src/models/saved",
        model_filename="lr_model_tfidf.pkl",
        encoder_save_path="src/encoders/saved",
        encoder_filename="tfidf.pkl"
    )

    # Neural Network model with GloVe encoder
    train_and_evaluate_model(
        model_type="neural_network",
        encoder_type="glove",
        data_path=dataset,
        model_save_path="src/models/saved",
        model_filename="nn_model_glove.pkl",
        encoder_save_path="src/encoders/saved",
        encoder_filename="glove.pkl"
    )

    # Neural Network model with TF-IDF encoder
    train_and_evaluate_model(
        model_type="neural_network",
        encoder_type="tfidf",
        data_path=dataset,
        model_save_path="src/models/saved",
        model_filename="nn_model_tfidf.pkl",
        encoder_save_path="src/encoders/saved",
        encoder_filename="tfidf.pkl"
    )


if __name__ == "__main__":
    # main()
    uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)

    # source venv/bin/activate
    # python -m src.main