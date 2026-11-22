"""Error analysis on the test set."""
import numpy as np
import pandas as pd

from .preprocess import NEGATIONS, normalize, word_count


def error_frame(texts: pd.Series, y_true, y_pred, y_score) -> pd.DataFrame:
    """One row per test review with prediction details."""
    out = pd.DataFrame({
        "text": texts.to_numpy(),
        "label": np.asarray(y_true),
        "pred": np.asarray(y_pred),
        "score": np.asarray(y_score),
    })
    out["correct"] = out["label"] == out["pred"]
    out["words"] = out["text"].map(word_count)
    return out
