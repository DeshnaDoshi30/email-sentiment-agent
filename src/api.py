# src/api.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.predict import SentimentPredictor

# -----------------------------
# APP INIT
# -----------------------------
app = FastAPI(
    title="Email Sentiment API",
    description="Predicts sentiment of emails using BERT",
    version="1.0"
)

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
predictor = SentimentPredictor()

# -----------------------------
# REQUEST SCHEMA
# -----------------------------
class EmailRequest(BaseModel):
    texts: List[str]

# -----------------------------
# RESPONSE SCHEMA
# -----------------------------
class PredictionResponse(BaseModel):
    label: str
    confidence: float

# -----------------------------
# ENDPOINT
# -----------------------------
@app.post("/predict", response_model=List[PredictionResponse])
def predict_sentiment(request: EmailRequest):
    results = predictor.predict_with_confidence(request.texts)
    return results
