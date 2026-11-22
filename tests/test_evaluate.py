import pytest

from sentiment import evaluate


def test_classification_metrics():
    m = evaluate.classification_metrics([1, 1, 0, 0], [1, 0, 0, 0], [0.9, 0.4, 0.2, 0.1])
    assert m["accuracy"] == 0.75
    assert m["precision"] == 1.0
    assert m["recall"] == 0.5
    assert m["roc_auc"] == 1.0


def test_confusion_table():
    t = evaluate.confusion_table([1, 1, 0], [1, 0, 0])
    assert t.loc["true_pos", "pred_neg"] == 1
    assert t.values.sum() == 3


def test_bootstrap_ci_contains_point_estimate():
    y = [1, 0] * 200
    pred = [1, 0] * 180 + [0, 1] * 20
    lo, hi = evaluate.bootstrap_ci(y, pred, n=300)
    assert lo < 0.9 < hi


def test_select_best():
    scores = {"a": {"accuracy": 0.8}, "b": {"accuracy": 0.9}}
    assert evaluate.select_best(scores) == "b"
    with pytest.raises(KeyError):
        evaluate.select_best(scores, metric="f2")
