"""Figures for reports/figures.

python -m mlops.plots
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from mlops.config import FIGURES_DIR, TARGET, TRAIN_FILE  # noqa: E402


def plot_alcohol_by_quality(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="alcohol", by=TARGET, ax=ax)
    ax.set_title("Alcohol by quality")
    fig.suptitle("")
    ax.set_xlabel("quality")
    ax.set_ylabel("alcohol (%)")
    return fig


def main() -> None:
    df = pd.read_csv(TRAIN_FILE)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "alcohol_by_quality.png"
    plot_alcohol_by_quality(df).savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: reports/figures/{out_path.name}")


if __name__ == "__main__":
    main()
