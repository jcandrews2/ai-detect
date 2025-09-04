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

# Define the model and encoder types
class ModelType(str, Enum):
    random_forest = "random_forest"
    logistic_regression = "logistic_regression"
    neural_network = "neural_network"

class EncoderType(str, Enum):
    tfidf = "tfidf"
    glove = "glove"

# Define the text item model
class TextItem(BaseModel): 
    text: str
    encoder: EncoderType = EncoderType.glove  # Default to GloVe
    model: ModelType = ModelType.logistic_regression  # Default to Logistic Regression
    
# Define some global variables
API_NAME = "AI-Generated Text Detection API"
API_DESCRIPTION = "Classifies text snippets as human or AI-generated."
API_VERSION = "1.0.0"

ENDPOINTS = {
    "info": "/api/v1/info",
    "classify": "/api/v1/classify",
    "health": "/api/v1/health"
}

ENCODERS = {
    EncoderType.tfidf: {
        "name": "TF-IDF",
        "class": TfidfEncoder,
        "should_fit": True,
        "description": "Term Frequency-Inverse Document Frequency"
    },
    EncoderType.glove: {
        "name": "GloVe",
        "class": GloveEncoder,
        "should_fit": False,
        "description": "Global Vectors for Word Representation"
    }
}

MODELS = {
    ModelType.random_forest: {
        "name": "Random Forest",
        "filename": "rf_model.pkl",
        "class": RandomForestModel,
        "description": "Ensemble learning model with high accuracy"
    },
    ModelType.logistic_regression: {
        "name": "Logistic Regression",
        "filename": "lr_model.pkl",
        "class": LogisticRegressionModel,
        "description": "Fast, interpretable linear model"
    },
    ModelType.neural_network: {
        "name": "Neural Network",
        "filename": "nn_model.pkl",
        "class": NeuralNetworkModel,
        "description": "Deep learning model with non-linear capabilities"
    }
}

# Define the model info
MODEL_INFO = {
    "available_models": [
        {
            "id": model_type.value,
            "name": info["name"],
            "description": info["description"]
        }
        for model_type, info in MODELS.items()
    ],
    "trained_on": "Kaggle Human vs. AI Generated Essays Dataset",
    "confidence_range": "0-100%"
}

CAPABILITIES = {
    "max_text_length": 5000,
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
        "capabilities": CAPABILITIES,
        "maintainer": MAINTAINER
    }

# Define the health endpoint
@api_router.get("/health")
async def health():
    return {}

# Define the classify endpoint
@api_router.post("/classify")
async def classify(text_item: TextItem):
    model_name = text_item.model
    encoder_name = text_item.encoder

    try:
        # Get model and encoder info
        model_info = MODELS[text_item.model]
        encoder_info = ENCODERS[text_item.encoder]
        
        # Construct model filename with encoder type
        model_filename = f"{text_item.model}_{text_item.encoder}_model.pkl"
        
        # Get absolute path to the model file
        model_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "src", "models", "saved")
        model_path = os.path.join(model_folder, model_filename)

        # Create encoder instance
        encoder = encoder_info["class"]()
        
        # Create model instance with encoder
        model = model_info["class"](encoder, encoder_info["should_fit"])
        
        # Load the model
        loaded_model = model.load(model_folder, model_filename)

        # Predict the class and probability
        pred, prob = loaded_model.predict(text_item.text)

        prediction = "AI" if pred[0] == 1 else "Human"
        
        return {
            "model": model_info["name"],
            "encoder": model_info["encoder"],
            "prediction": prediction,
            "confidence": {
                "human": f"{prob[0][0] * 100:.2f}%",
                "ai": f"{prob[0][1] * 100:.2f}%"
            }
        }

    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=f"Model {model_name} with {encoder_name} encoder not found. Please try again."
        )


app.include_router(api_router)

