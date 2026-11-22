"""Text normalisation. Kept deliberately light: TF-IDF works well on near-raw text,
and aggressive cleaning (stop-word removal, stemming) tends to drop signal such
as "not" and "no"."""
import html
import re

TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
SPACE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove tags such as <br /> and decode entities like &amp;."""
    return html.unescape(TAG.sub(" ", text))


def normalize(text: str) -> str:
    """Lower-case, drop URLs, separate punctuation and squeeze whitespace."""
    text = strip_html(text).lower()
    text = URL.sub(" ", text)
    text = re.sub(r"([!?.,;:()\"])", r" \1 ", text)
    text = text.replace("n't", " not")
    return SPACE.sub(" ", text).strip()
