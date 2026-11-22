"""Train all models, select on validation, evaluate once on test, write reports.

    PYTHONPATH=src python -m sentiment.pipeline
"""
import json
import time

import joblib
import pandas as pd

from . import errors, evaluate, explain, plots
from .config import FIGURES_DIR, MODELS_DIR, TABLES_DIR
from .io import drop_duplicates, load_reviews
from .models import all_models, scores, tfidf_logreg
from .preprocess import word_count
from .split import train_val_test


def compare_models(train: pd.DataFrame, val: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, fitted = {}, {}
    for name, model in all_models().items():
        t0 = time.time()
        model.fit(train["text"], train["label"])
        pred = model.predict(val["text"])
        rows[name] = evaluate.classification_metrics(val["label"], pred, scores(model, val["text"]))
        rows[name]["train_seconds"] = round(time.time() - t0, 1)
        fitted[name] = model
        print(f"{name}: {rows[name]}")
    return pd.DataFrame(rows).T.sort_values("accuracy", ascending=False), fitted


def run() -> dict:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_reviews()
    df = drop_duplicates(raw)
    train, val, test = train_val_test(df)
    data = {"raw_reviews": len(raw), "after_dedup": len(df),
            "train": len(train), "val": len(val), "test": len(test)}
    plots.length_hist(df["text"].map(word_count), df["label"], FIGURES_DIR / "review_length.png")

    val_table, fitted = compare_models(train, val)
    val_table.to_csv(TABLES_DIR / "validation_metrics.csv")
    plots.model_comparison(val_table, FIGURES_DIR / "model_comparison.png")
