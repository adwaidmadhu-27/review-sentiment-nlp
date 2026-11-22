"""Load reviews and labels."""
from pathlib import Path

import pandas as pd

from .config import RAW_CSV

LABELS = {"negative": 0, "positive": 1}


def load_reviews(path: Path | str = RAW_CSV) -> pd.DataFrame:
    """Columns: text, label (1 = positive)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run scripts/download_data.py")
    df = pd.read_csv(path)
    df = df.rename(columns={"review": "text", "sentiment": "label"})
    df["label"] = df["label"].map(LABELS)
    if df["label"].isna().any():
        raise ValueError("unexpected sentiment values")
    df["label"] = df["label"].astype(int)
    return df
