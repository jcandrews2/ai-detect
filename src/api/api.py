from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

app = FastAPI()

# Create a router with the prefix
api_router = APIRouter(prefix="/api/v1")

# uvicorn api:app --reload
class TextItem(BaseModel): 
    text: str

# Define some global variables
API_NAME = "AI Detection API"
API_DESCRIPTION = "Classifies text snippets as human or AI-generated"
API_VERSION = "1.0.0"

ENDPOINTS = {
    "info": "/api/v1/info",
    "classify": "/api/v1/classify",
    "health": "/api/v1/health"
}

MODEL_INFO = {
    "name": "Random Forest",
    "trained_on": "Kaggle Human vs. AI Generated Essays Dataset",
    "confidence_range": [0.0, 1.0]
}

CAPABILITIES = {
    "max_text_length": 5000,
    "batch_supported": False,
}

MAINTAINER = {
    "name": "Jimmy Andrews",
    "email": "jcandrews2@icloud.com"
}

@api_router.get("/")
async def root():
    return {"name": API_NAME, "version": API_VERSION, "endpoints": ENDPOINTS}

@api_router.get("/info")
async def info():
    return {
        "description": API_DESCRIPTION,
        "model": MODEL_INFO,
        "capabilities": CAPABILITIES,
        "maintainer": MAINTAINER
    }

@api_router.get("/health")
async def health():
    return {}

@api_router.get("/classify")
async def classify():
    return {}


app.include_router(api_router)

