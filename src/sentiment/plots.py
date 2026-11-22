"""Report figures."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, RED, GREY = "#2a6f97", "#c44536", "#8d99ae"


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def length_hist(words: pd.Series, labels: pd.Series, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for lab, color, name in ((1, BLUE, "positive"), (0, RED, "negative")):
        ax.hist(words[labels == lab].clip(upper=1000), bins=60, alpha=0.55, color=color, label=name)
    ax.set_xlabel("Words per review (clipped at 1,000)")
    ax.set_ylabel("Reviews")
    ax.set_title("Review length")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def model_comparison(metrics: pd.DataFrame, path: Path) -> Path:
    m = metrics.sort_values("accuracy")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(m.index, m["accuracy"] * 100, color=[GREY] * (len(m) - 1) + [BLUE])
    for i, v in enumerate(m["accuracy"] * 100):
        ax.text(v, i, f" {v:.2f}%", va="center")
    ax.set_xlim(80, 95)
    ax.set_xlabel("Validation accuracy (%)")
    ax.set_title("Model comparison")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def top_terms_chart(terms: pd.DataFrame, path: Path, n: int = 15) -> Path:
    pos = terms[terms["direction"] == "positive"].head(n).iloc[::-1]
    neg = terms[terms["direction"] == "negative"].head(n).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].barh(neg["term"], neg["weight"], color=RED)
    axes[0].set_title("Most negative terms")
    axes[1].barh(pos["term"], pos["weight"], color=BLUE)
    axes[1].set_title("Most positive terms")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)


def accuracy_by_length_chart(table: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [str(i) for i in table.index]
    ax.bar(labels, table["accuracy"] * 100, color=BLUE)
    for i, v in enumerate(table["accuracy"] * 100):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    ax.set_ylim(80, 100)
    ax.set_xlabel("Review length (words)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy by review length")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, path)
