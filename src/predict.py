# src/predict.py

import torch 
import torch.nn.functional as F 

from transformers import AutoTokenizer, AutoModelForSequenceClassification 

from src.config import MODEL_PATH, DEVICE, LABEL_MAPPING 


class SentimentPredictor:
    """
    Handles loading the model and predicting sentiment.
    """

    def __init__(self):
        self.device = torch.device(DEVICE)

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, texts: list[str]) -> list[str]:
        """
        Predict sentiment labels for a list of texts.
        """

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            predictions = torch.argmax(outputs.logits, dim=1)

        return [LABEL_MAPPING[int(label)] for label in predictions]
    
    def predict_with_confidence(self, texts: list[str]) -> list[dict]:
        """
        Predict sentiment with confidence scores.
        """

        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )

        encodings = {k: v.to(self.device) for k, v in encodings.items()}

        with torch.no_grad():
            outputs = self.model(**encodings)
            probs = F.softmax(outputs.logits, dim=1)

        results = []
        for prob_vector in probs:
            confidence, label_id = torch.max(prob_vector, dim=0)
            results.append({
                "label": LABEL_MAPPING[int(label_id)],
                "confidence": float(confidence)
            })

        return results
    



    
