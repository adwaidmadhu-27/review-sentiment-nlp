"""Evaluation metrics and model selection."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    if y_score is not None:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
    return {k: round(float(v), 4) for k, v in out.items()}


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return pd.DataFrame(m, index=["true_neg", "true_pos"], columns=["pred_neg", "pred_pos"])


def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Bootstrap confidence interval for accuracy - shows whether the gap
    between two models is bigger than test-set noise."""
    rng = np.random.default_rng(seed)
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    stats = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(n)]
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def select_best(val_scores: dict[str, dict], metric: str = "accuracy") -> str:
    """Pick the model with the best validation score (never the test score)."""
    return max(val_scores, key=lambda name: val_scores[name][metric])
