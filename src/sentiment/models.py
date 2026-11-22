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


class NBLogisticRegression(BaseEstimator, ClassifierMixin):
    """NB-LR (Wang & Manning, 2012): scale features by the Naive Bayes
    log-count ratio, then fit logistic regression. A strong classic baseline
    for sentiment."""

    def __init__(self, C: float = 4.0, alpha: float = 1.0):
        self.C = C
        self.alpha = alpha

    def fit(self, X, y):
        y = np.asarray(y)
        p = self.alpha + X[y == 1].sum(axis=0)
        q = self.alpha + X[y == 0].sum(axis=0)
        self.r_ = np.asarray(np.log((p / p.sum()) / (q / q.sum()))).ravel()
        self.lr_ = LogisticRegression(C=self.C, max_iter=2000, random_state=SEED)
        self.lr_.fit(self._scale(X), y)
        self.classes_ = self.lr_.classes_
        return self

    def _scale(self, X):
        return sparse.csr_matrix(X).multiply(self.r_).tocsr()

    def predict(self, X):
        return self.lr_.predict(self._scale(X))

    def predict_proba(self, X):
        return self.lr_.predict_proba(self._scale(X))

    def decision_function(self, X):
        return self.lr_.decision_function(self._scale(X))


def nb_logreg() -> Pipeline:
    return Pipeline([("vec", word_tfidf()), ("clf", NBLogisticRegression())])


def all_models() -> dict:
    return {
        "bow_naive_bayes": bag_of_words_nb(),
        "tfidf_logreg": tfidf_logreg(),
        "nb_logreg": nb_logreg(),
        "tfidf_word_char_svm": tfidf_svm(),
    }


def scores(model, texts) -> np.ndarray:
    """Continuous score for ranking metrics (probability or SVM margin)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)
