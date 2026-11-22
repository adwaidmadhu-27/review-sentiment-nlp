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
