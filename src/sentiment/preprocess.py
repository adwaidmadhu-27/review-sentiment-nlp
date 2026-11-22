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
