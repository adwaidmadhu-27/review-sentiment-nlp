"""Interpretation of linear text models."""
import numpy as np
import pandas as pd


def top_terms(pipeline, n: int = 20) -> pd.DataFrame:
    """Most positive and most negative n-grams by logistic-regression weight."""
    vec = pipeline.named_steps["vec"]
    clf = pipeline.named_steps["clf"]
    coef = clf.coef_.ravel()
    terms = np.asarray(vec.get_feature_names_out())
    order = np.argsort(coef)
    neg = pd.DataFrame({"term": terms[order[:n]], "weight": coef[order[:n]]})
    pos = pd.DataFrame({"term": terms[order[::-1][:n]], "weight": coef[order[::-1][:n]]})
    return pd.concat([pos.assign(direction="positive"), neg.assign(direction="negative")],
                     ignore_index=True)


def explain_text(pipeline, text: str, n: int = 8) -> pd.DataFrame:
    """Terms in one review that pushed the prediction most, with contribution =
    tf-idf value x weight."""
    vec = pipeline.named_steps["vec"]
    coef = pipeline.named_steps["clf"].coef_.ravel()
    x = vec.transform([text]).tocoo()
    terms = vec.get_feature_names_out()
    contrib = pd.DataFrame({"term": terms[x.col], "contribution": x.data * coef[x.col]})
    contrib["abs"] = contrib["contribution"].abs()
    return contrib.nlargest(n, "abs").drop(columns="abs").reset_index(drop=True)
