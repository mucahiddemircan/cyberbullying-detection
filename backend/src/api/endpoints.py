from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from ..services.predictor import predictor

router = APIRouter()

class PredictionRequest(BaseModel):
    """Schema for prediction requests."""
    text: str
    model_name: str

class PredictionResponse(BaseModel):
    """Schema for prediction responses."""
    prediction: str
    prediction_index: int
    probabilities: Dict[str, float]
    model_used: str
    vectorizer_used: str

@router.get("/models", response_model=List[str])
async def get_models():
    """Returns a list of all available machine learning models."""
    return predictor.get_available_models()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predicts the cyberbullying category for the given text using the specified model.
    """
    try:
        result = predictor.predict(request.text, request.model_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
