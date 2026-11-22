import pandas as pd

from sentiment import errors


def _errs():
    texts = pd.Series(["good film " * 60, "bad but fine", "not great", "awful"])
    return errors.error_frame(texts, [1, 0, 0, 0], [1, 1, 0, 1], [0.9, 0.6, 0.3, 0.95])


def test_error_frame():
    e = _errs()
    assert e["correct"].tolist() == [True, False, True, False]
    assert e["words"].iloc[0] == 120


def test_accuracy_by_length():
    t = errors.accuracy_by_length(_errs())
    assert t["reviews"].sum() == 4


def test_accuracy_by_flags():
    t = errors.accuracy_by_flags(_errs())
    neg = t[(t["flag"] == "has_negation") & t["value"]]
    assert neg["reviews"].iat[0] == 1
    assert errors.has_contrast("Bad, but fine")
