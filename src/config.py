# -----------------------------
# MODEL CONFIGURATION
# -----------------------------
MODEL_PATH = "models/bert/distilbert_email_model"
MODEL_NAME = "distilbert-email-sentiment-v1"
DEVICE = "cpu"  # change to "cuda" if GPU is available

CONFIDENCE_THRESHOLD = 0.70
LABEL_COLUMN = "sentiment"


# -----------------------------
# SENTIMENT LABELS
# -----------------------------
LABEL_MAPPING = {
    0: "negative",
    1: "neutral",
    2: "positive"
}


# -----------------------------
# DATA CONFIGURATION
# -----------------------------
TEXT_COLUMN = "email_text"
OUTPUT_COLUMN = "predicted_sentiment"


# -----------------------------
# FILE PATHS (DEFAULTS)
# -----------------------------
DEFAULT_INPUT_PATH = "data/raw/emails.xlsx"
DEFAULT_OUTPUT_PATH = "outputs/email_sentiment_results.xlsx"
