import numpy as np
import pytest

from sentiment import models


def test_bow_naive_bayes_learns_toy_data(reviews):
    m = models.bag_of_words_nb().fit(reviews["text"], reviews["label"])
    assert m.predict(["a wonderful and brilliant film"])[0] == 1


def test_tfidf_logreg_handles_negation(reviews):
    m = models.tfidf_logreg().fit(reviews["text"], reviews["label"])
    assert m.predict(["truly excellent and great"])[0] == 1
    assert m.predict(["truly awful and boring"])[0] == 0
    vocab = m.named_steps["vec"].vocabulary_
    assert any(t.startswith("NOT_") for t in vocab)


def test_tfidf_svm(reviews):
    m = models.tfidf_svm().fit(reviews["text"], reviews["label"])
    assert (m.predict(reviews["text"]) == reviews["label"]).mean() == 1


def test_nb_logreg_ratio_sign(reviews):
    m = models.nb_logreg().fit(reviews["text"], reviews["label"])
    vocab = m.named_steps["vec"].vocabulary_
    r = m.named_steps["clf"].r_
    assert r[vocab["wonderful"]] > 0 > r[vocab["awful"]]
    proba = m.predict_proba(["wonderful"])
    assert proba.shape == (1, 2)
