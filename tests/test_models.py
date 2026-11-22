import numpy as np
import pytest

from sentiment import models


def test_bow_naive_bayes_learns_toy_data(reviews):
    m = models.bag_of_words_nb().fit(reviews["text"], reviews["label"])
    assert m.predict(["a wonderful and brilliant film"])[0] == 1
