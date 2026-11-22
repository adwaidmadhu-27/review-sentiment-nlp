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


def word_tfidf(ngram_range=(1, 2), min_df: int = 2, max_features: int = 200_000) -> TfidfVectorizer:
    return TfidfVectorizer(preprocessor=clean, ngram_range=ngram_range, min_df=min_df,
                           max_features=max_features, sublinear_tf=True, token_pattern=r"\S+")


def tfidf_logreg(C: float = 4.0) -> Pipeline:
    """Word uni+bigram TF-IDF with negation marking + logistic regression."""
    return Pipeline([
        ("vec", word_tfidf()),
        ("clf", LogisticRegression(C=C, max_iter=2000, random_state=SEED)),
    ])


def tfidf_svm(C: float = 0.5) -> Pipeline:
    """Word + character n-grams + linear SVM. Char n-grams catch misspellings
    ("borrring") and word pieces."""
    feats = FeatureUnion([
        ("word", word_tfidf()),
        ("char", TfidfVectorizer(preprocessor=partial(clean, negation=False), analyzer="char_wb",
                                 ngram_range=(3, 5), min_df=5, max_features=200_000,
                                 sublinear_tf=True)),
    ])
    return Pipeline([("vec", feats), ("clf", LinearSVC(C=C, random_state=SEED))])
