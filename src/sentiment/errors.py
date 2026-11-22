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


def accuracy_by_length(errs: pd.DataFrame, bins=(0, 100, 200, 400, 800, 10_000)) -> pd.DataFrame:
    """Short reviews carry less evidence - check how accuracy changes with length."""
    bucket = pd.cut(errs["words"], bins=list(bins), right=False)
    g = errs.groupby(bucket, observed=True)["correct"]
    return pd.DataFrame({"reviews": g.size(), "accuracy": g.mean().round(4)})


def has_contrast(text: str) -> bool:
    """Reviews that turn around mid-way ("... but ...", "however") are harder."""
    t = normalize(text)
    return any(w in t.split() for w in ("but", "however", "although", "though"))


def accuracy_by_flags(errs: pd.DataFrame) -> pd.DataFrame:
    norm = errs["text"].map(normalize).str.split()
    flags = {
        "has_negation": norm.map(lambda ws: any(w in NEGATIONS for w in ws)),
        "has_contrast": errs["text"].map(has_contrast),
    }
    rows = []
    for name, f in flags.items():
        for val in (True, False):
            sub = errs[f == val]
            rows.append({"flag": name, "value": val, "reviews": len(sub),
                         "accuracy": round(float(sub["correct"].mean()), 4)})
    return pd.DataFrame(rows)


def most_confident_mistakes(errs: pd.DataFrame, n: int = 5, chars: int = 300) -> pd.DataFrame:
    """Wrong predictions with the most extreme scores - often label noise or sarcasm."""
    wrong = errs[~errs["correct"]].copy()
    # probabilities are centred on 0.5, SVM margins on 0
    centre = 0.5 if errs["score"].between(0, 1).all() else 0.0
    wrong["confidence"] = (wrong["score"] - centre).abs()
    top = wrong.nlargest(n, "confidence")
    top["snippet"] = top["text"].str.replace(r"<[^>]+>", " ", regex=True).str[:chars]
    return top[["label", "score", "words", "snippet"]].reset_index(drop=True)
