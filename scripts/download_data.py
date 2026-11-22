"""Download the IMDB 50k movie review dataset (Maas et al., 2011) as one CSV.

Original: Large Movie Review Dataset, https://ai.stanford.edu/~amaas/data/sentiment/
CSV mirror with columns `review`, `sentiment` (positive/negative).
"""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "imdb.csv"


def main() -> None:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"already downloaded: {DEST}")
        return
    print(f"downloading {URL} (~66 MB)")
    urlretrieve(URL, DEST)
    print(f"saved {DEST}")


if __name__ == "__main__":
    main()
