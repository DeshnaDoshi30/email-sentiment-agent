# src/mock_generator.py

import pandas as pd
import random
import time
from datetime import datetime
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INBOX_DIR = Path("data/inbox")
EMAILS_PER_BATCH = 5
SLEEP_TIME_SECONDS = 600  # 10 minutes

INBOX_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# MOCK EMAIL CONTENT (50+)
# -----------------------------
POSITIVE_EMAILS = [
    "Thank you for the quick resolution. Really appreciate the support.",
    "Great work on the recent update. Everything looks good.",
    "I am happy with the service provided. Keep it up!",
    "Excellent coordination by the team. Well done.",
    "The issue was resolved faster than expected. Thanks!",
    "Appreciate the proactive communication.",
    "This solution works perfectly for us.",
    "Very satisfied with the outcome.",
    "Thanks for addressing this so efficiently.",
    "Everything is running smoothly now.",
    "Good job handling this request.",
    "The turnaround time was impressive.",
    "Really happy with how this was managed.",
    "Support team did a fantastic job.",
    "No further issues from our side."
]

NEGATIVE_EMAILS = [
    "This issue is unacceptable and needs immediate attention.",
    "I am extremely disappointed with the delay.",
    "The problem still persists. This is frustrating.",
    "This has not been resolved despite multiple follow-ups.",
    "Very unhappy with the response time.",
    "The service quality has dropped significantly.",
    "This mistake caused major inconvenience.",
    "We expected much better handling of this issue.",
    "No proper update has been shared yet.",
    "This delay is impacting our operations.",
    "Unacceptable performance from the team.",
    "This is not what we agreed upon.",
    "The issue keeps repeating.",
    "Still waiting for a proper resolution.",
    "Completely dissatisfied with the outcome."
]

NEUTRAL_EMAILS = [
    "Please schedule a meeting for next week.",
    "Kindly review the attached document.",
    "Let me know your availability tomorrow.",
    "Sharing the report for your reference.",
    "Please find the details below.",
    "Requesting an update on the status.",
    "Forwarding this for your information.",
    "Can we connect later today?",
    "Adding you to the email thread.",
    "Please confirm receipt of this email.",
    "This is a reminder for the upcoming deadline.",
    "Looping in the relevant stakeholders.",
    "Let us know if any clarification is needed.",
    "Following up on the previous discussion.",
    "Noted. Will take this forward."
]

ALL_EMAILS = POSITIVE_EMAILS + NEGATIVE_EMAILS + NEUTRAL_EMAILS

# -----------------------------
# GENERATOR LOGIC
# -----------------------------
def generate_emails(n: int) -> pd.DataFrame:
    emails = random.choices(ALL_EMAILS, k=n)
    return pd.DataFrame({"email_text": emails})


def main():
    print("Mock Email Generator started...")

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = INBOX_DIR / f"emails_{timestamp}.xlsx"

        df = generate_emails(EMAILS_PER_BATCH)
        df.to_excel(file_path, index=False)

        print(f"Generated {EMAILS_PER_BATCH} emails → {file_path.name}")

        time.sleep(SLEEP_TIME_SECONDS)


if __name__ == "__main__":
    main()
