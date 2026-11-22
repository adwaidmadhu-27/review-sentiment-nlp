"""Reproducible stratified train / validation / test split."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import SEED, TEST_SIZE, VAL_SIZE


def train_val_test(df: pd.DataFrame, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 72 / 8 / 20 split (test first, then validation from the rest)."""
    rest, test = train_test_split(df, test_size=TEST_SIZE, stratify=df["label"], random_state=seed)
    train, val = train_test_split(rest, test_size=VAL_SIZE, stratify=rest["label"], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
