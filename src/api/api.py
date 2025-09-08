from fastapi import FastAPI, APIRouter, HTTPException
from pydantic import BaseModel
from enum import Enum
from src.models.random_forest import RandomForestModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.neural_network import NeuralNetworkModel
from src.encoders.tfidf_encoder import TfidfEncoder
from src.encoders.glove_encoder import GloveEncoder
from pathlib import Path
import os

app = FastAPI()

# Create a router with the prefix
api_router = APIRouter(prefix="/api/v1")

# Define the text item model
class TextItem(BaseModel): 
    text: str
    encoder: str = "glove"  # Default to GloVe
    model: str = "logistic_regression"  # Default to Logistic Regression

    @property
    def model_info(self):
        """Get model info or raise error if invalid."""
        if self.model not in MODELS:
            raise ValueError(f"Invalid model type. Available models: {list(MODELS.keys())}")
        return MODELS[self.model]

# Define some global variables
API_NAME = "AI-Generated Text Detection API"
API_DESCRIPTION = "Classifies text snippets as human or AI-generated."
API_VERSION = "1.0.0"

ENDPOINTS = {
    "info": "/api/v1/info",
    "classify": "/api/v1/classify",
}

ENCODERS = {
    "tfidf": {
        "name": "TF-IDF",
        "class": TfidfEncoder,
        "should_fit": True,
        "description": "Term Frequency-Inverse Document Frequency."
    },
    "glove": {
        "name": "GloVe",
        "class": GloveEncoder,
        "should_fit": False,
        "description": "Global Vectors for Word Representation."
    }
}

MODELS = {
    "random_forest": {
        "name": "Random Forest",
        "prefix": "rf",
        "class": RandomForestModel,
        "description": "Ensemble learning model with high accuracy."
    },
    "logistic_regression": {
        "name": "Logistic Regression",
        "prefix": "lr",
        "class": LogisticRegressionModel,
        "description": "Fast, interpretable linear model."
    },
    "neural_network": {
        "name": "Neural Network",
        "prefix": "nn",
        "class": NeuralNetworkModel,
        "description": "Deep learning model with non-linear capabilities."
    }
}

# Define the model info
MODEL_INFO = {
    "available_models": [
        {
            "id": model_id,
            "name": info["name"],
            "description": info["description"]
        }
        for model_id, info in MODELS.items()
    ],
    "trained_on": "Kaggle Human vs. AI Generated Essays Dataset",
    "confidence_range": "0-100%"
}

USAGE_EXAMPLES = {
    "classify": {
        "endpoint": "/api/v1/classify",
        "method": "POST",
        "request_format": {
            "text": "string (required) - The text to classify",
            "encoder": "string (optional) - Either 'glove' or 'tfidf', defaults to 'glove'",
            "model": "string (optional) - One of 'logistic_regression', 'random_forest', or 'neural_network', defaults to 'logistic_regression'"
        },
        "example_request": {
            "text": "This is a sample text that you want to classify",
            "encoder": "glove",
            "model": "logistic_regression"
        },
        "example_response": {
            "model": "Logistic Regression",
            "encoder": "glove",
            "prediction": "Human",
            "confidence": {
                "human": "85.50%",
                "ai": "14.50%"
            }
        }
    }
}

MAINTAINER = {
    "name": "Jimmy Andrews",
    "email": "jcandrews2@icloud.com"
}

# Define the root endpoint
@api_router.get("/")
async def root():
    return {"name": API_NAME, "version": API_VERSION, "endpoints": ENDPOINTS}

# Define the info endpoint
@api_router.get("/info")
async def info():
    return {
        "description": API_DESCRIPTION,
        "model": MODEL_INFO,
        "usage": USAGE_EXAMPLES,
        "maintainer": MAINTAINER
    }

# Define the classify endpoint
@api_router.post("/classify")
async def classify(text_item: TextItem):
    model_name = text_item.model
    encoder_name = text_item.encoder

    try:
        # Get model and encoder info
        model_info = text_item.model_info  # This will raise ValueError if model is invalid
        encoder_info = ENCODERS[text_item.encoder]
        
        # Construct model filename
        model_filename = f"{model_info['prefix']}_model_{text_item.encoder}.pkl"  # e.g. lr_model_glove.pkl

        # Load the model with the encoder
        model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "models", "saved")
        model_path = os.path.join(model_folder, model_filename)
        model_class = model_info["class"]
        model = model_class.load(model_folder, model_filename)

        # Predict the class and probability
        pred, prob = model.predict(text_item.text)

        # Convert prediction to string
        prediction = "AI" if pred[0] == 1 else "Human"
        
        return {
            "model": model_info["name"],
            "encoder": text_item.encoder,
            "prediction": prediction,
            "confidence": {
                "human": f"{prob[0][0] * 100:.2f}%",
                "ai": f"{prob[0][1] * 100:.2f}%"
            }
        }

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {e}"
        )


app.include_router(api_router)

