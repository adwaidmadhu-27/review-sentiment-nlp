"""Text classification pipelines."""
from functools import partial

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import SEED
from .preprocess import clean


def bag_of_words_nb() -> Pipeline:
    """Baseline: word counts + multinomial Naive Bayes."""
    return Pipeline([
        ("vec", CountVectorizer(preprocessor=partial(clean, negation=False), min_df=2)),
        ("clf", MultinomialNB()),
    ])
