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


NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor", "without"}
CLAUSE_END = {".", ",", "!", "?", ";", ":", "but"}


def mark_negation(text: str, window: int = 3) -> str:
    """Prefix up to `window` words after a negation with NOT_ ("not good" ->
    "not NOT_good"), so the model can learn that negated words flip meaning."""
    out, left = [], 0
    for tok in text.split():
        if tok in CLAUSE_END:
            left = 0
            out.append(tok)
        elif tok in NEGATIONS:
            left = window
            out.append(tok)
        elif left > 0:
            out.append("NOT_" + tok)
            left -= 1
        else:
            out.append(tok)
    return " ".join(out)


def clean(text: str, negation: bool = True) -> str:
    """Full preprocessing used by every model."""
    t = normalize(text)
    return mark_negation(t) if negation else t


def word_count(text: str) -> int:
    return len(strip_html(text).split())
