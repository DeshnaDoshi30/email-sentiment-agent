# src/agent_loop.py

import time
from pathlib import Path
import pandas as pd

from src.pipeline import run_pipeline
from src.config import TEXT_COLUMN
from src.storage import read_excel, save_predictions_to_excel

# -----------------------------
# CONFIG
# -----------------------------
INBOX_DIR = Path("data/inbox")
OUTPUT_DIR = Path("outputs/processed")
MEMORY_FILE = Path("outputs/processed_files.txt")

SLEEP_TIME_SECONDS = 1800  # 30 minutes

INBOX_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MEMORY_FILE.touch(exist_ok=True)


# -----------------------------
# MEMORY
# -----------------------------
def load_processed_files() -> set[str]:
    return set(MEMORY_FILE.read_text().splitlines())


def mark_as_processed(filename: str):
    with MEMORY_FILE.open("a") as f:
        f.write(filename + "\n")


# -----------------------------
# AGENT LOOP
# -----------------------------
def main():
    print("🤖 Email Sentiment Agent started")

    while True:
        processed = load_processed_files()
        inbox_files = list(INBOX_DIR.glob("*.xlsx"))

        new_files = [f for f in inbox_files if f.name not in processed]

        if not new_files:
            print("⏸️ No new files. Sleeping...")
        else:
            print(f"📬 Found {len(new_files)} new batch(es)")

        for file_path in new_files:
            print(f"🔍 Processing {file_path.name}")

            df = read_excel(file_path)

            if TEXT_COLUMN not in df.columns:
                print(f"⚠️ Missing column in {file_path.name}, skipping")
                mark_as_processed(file_path.name)
                continue

            emails = df[TEXT_COLUMN].tolist()

            results = run_pipeline(emails)

            results_df = pd.DataFrame(results)

            output_path = OUTPUT_DIR / file_path.name
            save_predictions_to_excel(results_df, output_path)

            mark_as_processed(file_path.name)
            print(f"✅ Saved → {output_path.name}")

        print("😴 Sleeping for 30 minutes...\n")
        time.sleep(SLEEP_TIME_SECONDS)


if __name__ == "__main__":
    main()
