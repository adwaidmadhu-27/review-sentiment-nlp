import pytest

from sentiment import explain
from sentiment.models import tfidf_logreg


@pytest.fixture
def lr(reviews):
    return tfidf_logreg().fit(reviews["text"], reviews["label"])


def test_top_terms(lr):
    t = explain.top_terms(lr, 5)
    assert len(t) == 10
    assert (t.loc[t["direction"] == "positive", "weight"] > 0).all()
    assert (t.loc[t["direction"] == "negative", "weight"] < 0).all()
