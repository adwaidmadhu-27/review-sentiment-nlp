import pandas as pd
import pytest

from sentiment import io as sio


def test_load_reviews_maps_labels(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame({"review": ["good", "bad"], "sentiment": ["positive", "negative"]}).to_csv(p, index=False)
    df = sio.load_reviews(p)
    assert list(df.columns) == ["text", "label"]
    assert df["label"].tolist() == [1, 0]


def test_load_reviews_rejects_unknown_label(tmp_path):
    p = tmp_path / "r.csv"
    pd.DataFrame({"review": ["meh"], "sentiment": ["neutral"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        sio.load_reviews(p)
