"""Score new reviews with the saved model.

    PYTHONPATH=src python -m sentiment.predict "One of the best films I have seen"
"""
import sys
from pathlib import Path

import joblib

from .config import MODELS_DIR
from .models import scores


def load_model(path: Path = MODELS_DIR / "best_model.joblib"):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `make train` first")
    return joblib.load(path)
