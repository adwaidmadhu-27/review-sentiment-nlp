import pandas as pd

from sentiment.split import train_val_test


def test_split_sizes_and_stratification():
    df = pd.DataFrame({"text": [f"r{i}" for i in range(1000)], "label": [0, 1] * 500})
    train, val, test = train_val_test(df)
    assert len(test) == 200 and len(val) == 80 and len(train) == 720
    for part in (train, val, test):
        assert abs(part["label"].mean() - 0.5) < 0.02
    assert not set(train["text"]) & set(test["text"])


def test_split_is_reproducible():
    df = pd.DataFrame({"text": [f"r{i}" for i in range(200)], "label": [0, 1] * 100})
    a = train_val_test(df)[2]["text"].tolist()
    b = train_val_test(df)[2]["text"].tolist()
    assert a == b
