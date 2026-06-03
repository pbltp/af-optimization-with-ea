import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


def generate_two_moons(n_samples=1000, noise=0.2, random_state=42):
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=["x", "y"])
    df["label"] = y.astype(int)
    return df


if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent

    df = generate_two_moons(n_samples=1000, noise=0.2, random_state=42)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    train_df.to_csv(output_dir / "two_moons_train.csv", index=False)
    test_df.to_csv(output_dir / "two_moons_test.csv", index=False)

    plt.figure(figsize=(8, 8))
    plt.scatter(
        df["x"],
        df["y"],
        c=df["label"],
        cmap="Spectral",
        edgecolors="k",
        alpha=0.7,
    )
    plt.title("Two Moons")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.axis("equal")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.savefig(output_dir / "two_moons.png")
