import pandas as pd
import pytest

POS = [
    "A wonderful film with great acting and a brilliant script.",
    "I loved it. One of the best movies this year, truly excellent!",
    "Great story, amazing cast<br /><br />highly recommended.",
    "Not bad at all, actually quite enjoyable and fun.",
    "Beautiful, moving and perfect from start to finish.",
    "An excellent and wonderful experience, I loved every minute.",
]
NEG = [
    "A terrible film with awful acting and a boring script.",
    "I hated it. One of the worst movies this year, truly bad!",
    "Boring story, poor cast<br /><br />a complete waste of time.",
    "Not good at all, actually quite dull and annoying.",
    "Ugly, pointless and horrible from start to finish.",
    "An awful and boring experience, I hated every minute.",
]


@pytest.fixture
def reviews() -> pd.DataFrame:
    texts = (POS + NEG) * 5
    labels = ([1] * len(POS) + [0] * len(NEG)) * 5
    return pd.DataFrame({"text": texts, "label": labels})
