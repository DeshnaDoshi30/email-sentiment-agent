# src/preprocess.py

import pandas as pd 


def clean_text(text: str) -> str:
    """
    Clean a single email text string.
    """
    if pd.isna(text):
        return ""

    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())  # remove extra spaces

    return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Apply text cleaning to a DataFrame.
    """
    df = df.copy()
    df[text_column] = df[text_column].apply(clean_text)
    return df
