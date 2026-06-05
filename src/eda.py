from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compute_eda_summary(ratings: pd.DataFrame) -> dict:
    n_users = ratings["user_id"].nunique()
    n_movies = ratings["movie_id"].nunique()
    n_ratings = len(ratings)
    possible = max(n_users * n_movies, 1)
    sparsity = 1.0 - (n_ratings / possible)
    return {
        "n_ratings": int(n_ratings),
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "rating_mean": float(ratings["rating"].mean()),
        "rating_median": float(ratings["rating"].median()),
        "sparsity": float(sparsity),
        "date_min": str(ratings["date"].min().date()) if "date" in ratings else None,
        "date_max": str(ratings["date"].max().date()) if "date" in ratings else None,
    }


def save_eda_figures(ratings: pd.DataFrame, output_dir: str | Path) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    ratings["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Rating Distribution")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = output_dir / "rating_distribution.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    top_movies = ratings.groupby(["movie_id", "title"], dropna=False).size().sort_values(ascending=False).head(10).sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [str(idx[1])[:35] for idx in top_movies.index]
    ax.barh(labels, top_movies.values)
    ax.set_title("Top Movies by Rating Count")
    ax.set_xlabel("Number of ratings")
    fig.tight_layout()
    path = output_dir / "top_movies.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    user_activity = ratings.groupby("user_id").size()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(user_activity, bins=min(20, max(5, user_activity.nunique())))
    ax.set_title("User Activity Distribution")
    ax.set_xlabel("Ratings per user")
    ax.set_ylabel("Users")
    fig.tight_layout()
    path = output_dir / "user_activity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    if "date" in ratings.columns and ratings["date"].notna().any():
        monthly = ratings.set_index("date").resample("ME").size()
        fig, ax = plt.subplots(figsize=(7, 4))
        monthly.plot(ax=ax)
        ax.set_title("Rating Activity Over Time")
        ax.set_xlabel("Date")
        ax.set_ylabel("Ratings")
        fig.tight_layout()
        path = output_dir / "ratings_over_time.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    return paths
