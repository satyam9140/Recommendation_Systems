from __future__ import annotations

import pandas as pd


def attach_titles(recs: pd.DataFrame, titles: pd.DataFrame) -> pd.DataFrame:
    if titles is None or titles.empty or "title" not in titles.columns:
        out = recs.copy()
        out["title"] = out["movie_id"].map(lambda x: f"Movie {x}")
        return out
    return recs.merge(titles, on="movie_id", how="left")


def generate_recommendations_for_users(model, users: list[int], train_df: pd.DataFrame, titles: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    rows = []
    all_movie_ids = sorted(train_df["movie_id"].unique().tolist())
    for user_id in users:
        recs = model.recommend(int(user_id), train_df, k=k, candidates=all_movie_ids)
        recs = attach_titles(recs, titles)
        recs.insert(0, "rank", range(1, len(recs) + 1))
        recs.insert(0, "user_id", int(user_id))
        rows.append(recs)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
