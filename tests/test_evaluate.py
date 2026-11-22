import pytest

from sentiment import evaluate


def test_classification_metrics():
    m = evaluate.classification_metrics([1, 1, 0, 0], [1, 0, 0, 0], [0.9, 0.4, 0.2, 0.1])
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["roc_auc"] == 1.0
