# =============================
# IMPORTS
# =============================
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

# =============================
# CONFIG
# =============================
DATA_PATH = "data/processed/email_sentiment_labeled.csv"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert_50k"

TEXT_COL = "clean_email"
LABEL_COL = "labels"

MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 2
SAMPLE_SIZE = 50_000
SEED = 42

# =============================
# LOAD + SAMPLE DATA
# =============================
df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COL, LABEL_COL]].dropna().reset_index(drop=True)

print("Full dataset:")
print(df[LABEL_COL].value_counts())

# 🔥 limit to 50k samples
df = df.sample(SAMPLE_SIZE, random_state=SEED).reset_index(drop=True)

print("\nUsing 50k samples:")
print(df[LABEL_COL].value_counts())

# =============================
# DATASET + TOKENIZATION
# =============================
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1, seed=SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch[TEXT_COL],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns([TEXT_COL])
dataset.set_format("torch")

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# =============================
# CLASS BALANCING
# =============================
labels = train_dataset["labels"]
label_tensor = torch.tensor(labels)

class_counts = torch.bincount(label_tensor)
class_weights = 1.0 / class_counts.float()
sample_weights = class_weights[label_tensor]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# =============================
# MODEL
# =============================
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3
)

# =============================
# TRAINING ARGUMENTS
# (old-version safe)
# =============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=100,
    save_strategy="no",
    report_to="none"
)

# =============================
# CUSTOM TRAINER
# =============================
class BalancedTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler
        )

# =============================
# TRAIN
# =============================
trainer = BalancedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()

# =============================
# SAVE MODEL
# =============================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Training finished successfully on 50k samples")
