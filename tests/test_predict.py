import pytest

from sentiment import predict
from sentiment.models import tfidf_logreg


def test_load_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "missing.joblib")
