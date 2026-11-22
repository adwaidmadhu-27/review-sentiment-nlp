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
