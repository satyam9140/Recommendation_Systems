from __future__ import annotations

import pandas as pd


def filter_interactions(
    ratings: pd.DataFrame,
    min_user_ratings: int = 3,
    min_movie_ratings: int = 3,
    max_rounds: int = 3,
) -> pd.DataFrame:
    """Iteratively filter very sparse users/items for stable small-scale training."""
    df = ratings.copy()
    for _ in range(max_rounds):
        before = len(df)
        if min_user_ratings > 1:
            user_counts = df["user_id"].value_counts()
            df = df[df["user_id"].isin(user_counts[user_counts >= min_user_ratings].index)]
        if min_movie_ratings > 1:
            movie_counts = df["movie_id"].value_counts()
            df = df[df["movie_id"].isin(movie_counts[movie_counts >= min_movie_ratings].index)]
        if len(df) == before:
            break
    return df.reset_index(drop=True)


def temporal_train_test_split(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    min_train_items: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """User-wise temporal split: newest interactions become test rows.

    This avoids random leakage across time and is closer to how recommenders are
    used in production: learn from the past, recommend future items.
    """
    train_parts = []
    test_parts = []
    for _, user_df in ratings.sort_values("date").groupby("user_id", sort=False):
        n = len(user_df)
        if n <= min_train_items:
            train_parts.append(user_df)
            continue
        n_test = max(1, int(round(n * test_ratio)))
        n_test = min(n_test, n - min_train_items)
        train_parts.append(user_df.iloc[:-n_test])
        test_parts.append(user_df.iloc[-n_test:])
    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=ratings.columns)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings.columns)
    return train, test


def make_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
