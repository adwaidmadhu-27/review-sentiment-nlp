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


def predict(texts: list[str], model=None) -> list[dict]:
    model = model or load_model()
    labels = model.predict(texts)
    s = scores(model, texts)
    return [{"text": t[:80], "sentiment": "positive" if lab == 1 else "negative", "score": round(float(v), 3)}
            for t, lab, v in zip(texts, labels, s)]


def main(argv: list[str]) -> None:
    if not argv:
        sys.exit('usage: python -m sentiment.predict "review text" ["another review" ...]')
    for row in predict(argv):
        print(f"{row['sentiment']:>8}  {row['score']:>7}  {row['text']}")


if __name__ == "__main__":
    main(sys.argv[1:])
