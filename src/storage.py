# src/storage.py

import pandas as pd # type: ignore
from pathlib import Path

def read_excel(file_path: Path | str) -> pd.DataFrame:
    """
    Read an Excel file and return a DataFrame.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_excel(file_path)

    
def save_predictions_to_excel(predictions: list[dict], output_path: str):
    """
    Save prediction results to an Excel file.
    """

    df = pd.DataFrame(predictions)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(output_path, index=False)
