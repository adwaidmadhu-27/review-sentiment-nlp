# Data

`python scripts/download_data.py` saves `data/raw/imdb.csv` (~66 MB).

- 50,000 movie reviews from IMDB, balanced: 25,000 positive and 25,000 negative.
- Columns: `review` (raw text, contains `<br />` HTML tags), `sentiment`
  (`positive` / `negative`).
- The original labels come from star ratings: <= 4/10 is negative,
  >= 7/10 is positive; neutral reviews are excluded.

Citation: Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment
Analysis*. ACL. https://ai.stanford.edu/~amaas/data/sentiment/
