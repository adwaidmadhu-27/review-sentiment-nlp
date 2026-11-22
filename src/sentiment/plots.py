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
