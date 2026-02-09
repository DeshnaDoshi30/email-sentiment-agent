# Email Sentiment Analysis Agent

An end-to-end NLP-based email sentiment analysis system built using **DistilBERT**, designed with a **confidence-based decision mechanism** and a **batch-processing agent loop**.  
This project focuses on building a production-style ML pipeline rather than just a standalone model.

---

## 🚀 Project Overview

The system analyzes incoming emails and classifies their sentiment (e.g., Positive, Neutral, Negative).  
Predictions are accompanied by confidence scores, which are used to decide whether an email can be **automatically classified** or should be **flagged for manual review**.

The architecture is intentionally designed to simulate real-world enterprise pipelines and can be extended to domains like **cybersecurity log analysis**.

---

## 🧠 Key Features

- Text preprocessing pipeline for raw emails
- Transformer-based sentiment classification (DistilBERT)
- Confidence-based decision logic (threshold-driven)
- Batch processing of emails at fixed intervals
- Agent loop for continuous monitoring
- Audit-friendly outputs with metadata
- Clean modular codebase

---

## 🏗️ System Design & Methodology

### 1. Initial Sentiment Bootstrapping
- VADER sentiment analysis was used initially for understanding sentiment distributions and validating label consistency.

### 2. Model Training
- DistilBERT was fine-tuned on labeled email data for multi-class sentiment classification.
- PyTorch is used internally via Hugging Face Transformers for:
  - Tensor operations
  - Loss computation
  - Backpropagation
  - Model optimization

### 3. Inference with Confidence
- Softmax probabilities are extracted from model logits.
- Confidence score = maximum class probability.
- Predictions are routed based on confidence threshold:
  - **≥ 90%** → Auto-classified
  - **< 90%** → Marked for manual review

### 4. Batch-Based Agent Loop
- Emails are processed in batches every **30 minutes**.
- Prevents continuous polling and reduces compute overhead.
- Simulates real-world scheduled ML pipelines.

---

## 📂 Project Structure

email-sentiment-agent/
- │
- ├── src/
│ ├── init.py
│ ├── agent_loop.py # Long-running batch agent
│ ├── pipeline.py # Core sentiment pipeline logic
│ ├── predict.py # Model loading & inference
│ ├── preprocess.py # Text cleaning utilities
│ ├── config.py # Central configuration
│ └── storage.py # File I/O helpers
│
├── data/
│ └── inbox/ # Incoming email batches
│
├── outputs/
│ └── processed/ # Processed results
│
├── models/ # Trained model (ignored in git)
│
├── requirements.txt
├── README.md
└── .gitignore

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/email-sentiment-agent.git
cd email-sentiment-agent

pip install -r requirements.txt

▶️ How to Run the Project
Run the Agent (recommended)

This starts the full system with batch processing:

python -m src.agent_loop

Run Pipeline Manually (for testing)
from src.pipeline import run_pipeline

emails = [
    "Thank you for the quick resolution!",
    "This issue is very frustrating and unresolved."
]

results = run_pipeline(emails)
print(results)

📊 Output Format

Each email produces a structured result:

{
  "email_text": "Original email text",
  "predicted_sentiment": "negative",
  "confidence": 0.93,
  "status": "auto_classified",
  "model_name": "distilbert_email_model",
  "created_at": "2026-02-09T10:30:00"
}

🔮 Future Improvements

- Confidence calibration (temperature scaling)
- Active learning from low-confidence samples
- Human-in-the-loop feedback system
- Real-time streaming mode

🧑‍💻 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- pandas
- NumPy
