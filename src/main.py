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


def select_encoder_and_model(
    encoder_type: Literal["tfidf", "glove"],
    model_type: Literal["random_forest", "logistic_regression", "neural_network"]
) -> tuple[TextEncoder, MLTextClassifier | NNTextClassifier]:
    """Select the encoder and model."""

    encoder_classes = {
        "tfidf": [TfidfEncoder, True],
        "glove": [GloveEncoder, False]
    }

    encoder_class, should_fit_encoder = encoder_classes[encoder_type]
    encoder = encoder_class()

    model_classes = {
        "random_forest": RandomForestModel,
        "logistic_regression": LogisticRegressionModel,
        "neural_network": NeuralNetworkModel
    }
    model_class = model_classes[model_type]
    model = model_class(encoder, should_fit_encoder)

    return encoder, model


def train_and_evaluate_ml_model(
    model_type: Literal["random_forest", "logistic_regression"],
    encoder_type: Literal["tfidf", "glove"],
    data_path: str,
    save_path: str,
    model_filename: str
) -> MLTextClassifier:
    """Train and evaluate an ML text detection model.
    
    Args:
        model_type: Type of model to use ("random_forest" or "logistic_regression")
        encoder_type: Type of encoder to use ("tfidf" or "glove")
        data_path: Path to the data CSV file
        save_path: Directory to save the trained model
        model_filename: Filename for the saved model
    
    Returns:
        The trained model instance
    """

    # Load and prepare the data
    print("===== Preparing data =====\n")
    data_loader = DataLoader(data_path)
    data_loader.prepare()
    
    X_train, y_train = data_loader.get_train_data()
    X_test, y_test = data_loader.get_test_data()

    # Select the encoder and model
    encoder, model = select_encoder_and_model(encoder_type, model_type)

    # Train and evaluate
    model.train(X_train, y_train)
    print()
    print("===== Evaluating model =====\n")
    model.evaluate(X_test, y_test)
    model.save(save_path, model_filename)

    # Test loading and using the saved model
    print("\n===== Testing saved model =====")
    loaded_model = model.load(save_path, model_filename)
    
    # Verify it works with an example prediction
    test_text = "The rise of artificial intelligence has brought both innovation and uncertainty to the field of written communication. As large language models become increasingly sophisticated, the distinction between human and machine writing grows more difficult to identify. Traditional machine learning methods, such as logistic regression or support vector machines, can still provide strong baselines for classification when paired with textual features like TF-IDF and n-grams. However, as AI output improves in fluency and variety, the boundaries blur, forcing researchers to rely on more subtle signals such as perplexity, stylistic variation, and contextual coherence. Detecting AI-generated text is not simply a technical challenge—it is also a societal one, raising questions about authorship, authenticity, and trust in digital information. Ultimately, the task demands both computational rigor and careful consideration of how these tools are applied in real-world settings."

    pred, prob = loaded_model.predict(test_text)
    print(f"\nTest prediction (1=AI, 0=Human): {pred[0]}")
    print(f"Confidence: {prob[0].max():.2%}")

    return loaded_model

def train_and_evaluate_nn_model(
    model_type: Literal["neural_network"],
    encoder_type: Literal["tfidf", "glove"],
    data_path: str,
    save_path: str,
    model_filename: str
) -> NNTextClassifier:
    """Train and evaluate an NN text detection model.
    
    Args:
        model_type: Type of model to use ("neural_network")
        encoder_type: Type of encoder to use ("tfidf" or "glove")
        data_path: Path to the data CSV file
        save_path: Directory to save the trained model
        model_filename: Filename for the saved model
    
    Returns:
        The trained model instance
    """

    # Load and prepare the data
    print("===== Preparing data =====\n")
    data_loader = DataLoader(data_path)
    data_loader.prepare()
    
    X_train, y_train = data_loader.get_train_data()
    X_test, y_test = data_loader.get_test_data()

    # Select the encoder and model
    encoder, model = select_encoder_and_model(encoder_type, model_type)

    # Train and evaluate
    model.train_model(X_train, y_train)
    print()
    print("===== Evaluating model =====\n")
    model.evaluate(X_test, y_test)
    model.save(save_path, model_filename)

    # Test loading and using the saved model
    print("\n===== Testing saved model =====")
    loaded_model = model.load(save_path, model_filename)
    
    # Verify it works with an example prediction
    test_text = "The rise of artificial intelligence has brought both innovation and uncertainty to the field of written communication. As large language models become increasingly sophisticated, the distinction between human and machine writing grows more difficult to identify. Traditional machine learning methods, such as logistic regression or support vector machines, can still provide strong baselines for classification when paired with textual features like TF-IDF and n-grams. However, as AI output improves in fluency and variety, the boundaries blur, forcing researchers to rely on more subtle signals such as perplexity, stylistic variation, and contextual coherence. Detecting AI-generated text is not simply a technical challenge—it is also a societal one, raising questions about authorship, authenticity, and trust in digital information. Ultimately, the task demands both computational rigor and careful consideration of how these tools are applied in real-world settings."

    pred, prob = loaded_model.predict(test_text)
    print(f"\nTest prediction (1=AI, 0=Human): {pred[0]}")
    print(f"Confidence: {prob[0].max():.2%}")

    return loaded_model

def main():
    # dataset = "~/Downloads/AI_Human.csv"
    dataset = "~/Downloads/balanced_ai_human_prompts.csv"

    # Random Forest model with GloVe encoder
    rf_model = train_and_evaluate_ml_model(
        model_type="random_forest",
        encoder_type="glove",
        data_path=dataset,
        save_path="src/models/saved",
        model_filename="rf_model_glove.pkl"
    )

    # Random Forest model with TF-IDF encoder
    rf_model = train_and_evaluate_ml_model(
        model_type="random_forest",
        encoder_type="glove",
        data_path=dataset,
        save_path="src/models/saved",
        model_filename="rf_model_tfidf.pkl"
    )

    # Logistic Regression model with GloVe encoder
    lr_model = train_and_evaluate_ml_model(
        model_type="logistic_regression",
        encoder_type="glove",
        data_path=dataset,
        save_path="src/models/saved",
        model_filename="lr_model_glove.pkl"
    )

    # Logistic Regression model with TF-IDF encoder
    lr_model = train_and_evaluate_ml_model(
        model_type="logistic_regression",
        encoder_type="glove",
        data_path=dataset,
        save_path="src/models/saved",
        model_filename="lr_model_tfidf.pkl"
    )

    # Neural Network model with GloVe encoder
    nn_model = train_and_evaluate_nn_model(
        model_type="neural_network",
        encoder_type="tfidf",
        data_path=dataset,
        save_path="src/models/saved",
        model_filename="nn_model_glove.pkl"
    )

    # Neural Network model with TF-IDF encoder
    nn_model = train_and_evaluate_nn_model(
        model_type="neural_network",
        encoder_type="tfidf",
        data_path=dataset,
        save_path="src/models/saved",
        model_filename="nn_model_tfidf.pkl"
    )


if __name__ == "__main__":
    main()
    # uvicorn.run("src.api.api:app", host="0.0.0.0", port=8000, reload=True)

    # source venv/bin/activate
    # python -m src.main