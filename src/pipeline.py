# src/pipeline.py

from datetime import datetime
from typing import List, Dict

from src.predict import SentimentPredictor
from src.preprocess import clean_text
from src.config import MODEL_NAME, CONFIDENCE_THRESHOLD


def run_pipeline(emails: List[str]) -> List[Dict]:
    """
    Core sentiment analysis pipeline.
    Takes raw emails → returns structured predictions with confidence logic.
    """

    # 1. Clean text
    cleaned_emails = [clean_text(email) for email in emails]

    # 2. Load predictor once
    predictor = SentimentPredictor()

    predictions = predictor.predict_with_confidence(cleaned_emails)

    # 3. Apply confidence-based decisioning
    results = []

    for raw_text, pred in zip(emails, predictions):
        confidence = pred["confidence"]
        label = pred["label"]

        status = (
            "auto_classified"
            if confidence >= CONFIDENCE_THRESHOLD
            else "needs_manual_review"
        )

        results.append({
            "email_text": raw_text,
            "predicted_sentiment": label,
            "confidence": round(confidence, 4),
            "status": status,
            "model_name": MODEL_NAME,
            "processed_at": datetime.utcnow().isoformat()
        })

    return results
