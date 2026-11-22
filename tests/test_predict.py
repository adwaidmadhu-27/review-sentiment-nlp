import pytest

from sentiment import predict
from sentiment.models import tfidf_logreg


def test_load_model_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        predict.load_model(tmp_path / "missing.joblib")


def test_predict_with_model(reviews):
    m = tfidf_logreg().fit(reviews["text"], reviews["label"])
    out = predict.predict(["an excellent and wonderful film", "awful and boring"], model=m)
    assert [r["sentiment"] for r in out] == ["positive", "negative"]
    assert 0 <= out[0]["score"] <= 1


def test_main_without_args_exits():
    with pytest.raises(SystemExit):
        predict.main([])
