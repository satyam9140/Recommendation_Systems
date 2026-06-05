from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCFRecommender:
    """Item-based collaborative filtering using cosine similarity.

    The model recommends movies similar to the items a user has already rated.
    It is explainable and works well as a practical baseline on sparse datasets.
    """

    name = "item_item_cf"

    def __init__(self, top_n_neighbors: int = 30, min_similarity: float = 0.0):
        self.top_n_neighbors = top_n_neighbors
        self.min_similarity = min_similarity
        self.global_mean = 3.0
        self.user_means: pd.Series | None = None
        self.item_means: pd.Series | None = None
        self.movie_ids: list[int] = []
        self.similarity: pd.DataFrame | None = None
        self.user_ratings: dict[int, dict[int, float]] = {}

    def fit(self, ratings: pd.DataFrame):
        self.global_mean = float(ratings["rating"].mean())
        self.movie_ids = sorted(ratings["movie_id"].unique().tolist())
        self.user_means = ratings.groupby("user_id")["rating"].mean()
        self.item_means = ratings.groupby("movie_id")["rating"].mean()
        matrix = ratings.pivot_table(index="user_id", columns="movie_id", values="rating")
        centered = matrix.sub(matrix.mean(axis=1), axis=0).fillna(0.0)
        sim = cosine_similarity(centered.T)
        self.similarity = pd.DataFrame(sim, index=centered.columns, columns=centered.columns)
        self.user_ratings = {
            int(uid): dict(zip(g["movie_id"].astype(int), g["rating"].astype(float)))
            for uid, g in ratings.groupby("user_id")
        }
        return self

    def predict_one(self, user_id: int, movie_id: int) -> float:
        if self.similarity is None:
            raise RuntimeError("Model is not fitted.")
        rated = self.user_ratings.get(user_id, {})
        if not rated:
            return float(np.clip(self.item_means.get(movie_id, self.global_mean), 1, 5))
        if movie_id not in self.similarity.index:
            return float(np.clip(self.user_means.get(user_id, self.global_mean), 1, 5))

        sims = []
        for other_movie, rating in rated.items():
            if other_movie == movie_id or other_movie not in self.similarity.columns:
                continue
            sim = float(self.similarity.loc[movie_id, other_movie])
            if sim > self.min_similarity:
                sims.append((sim, rating))
        if not sims:
            base = 0.5 * self.user_means.get(user_id, self.global_mean) + 0.5 * self.item_means.get(movie_id, self.global_mean)
            return float(np.clip(base, 1, 5))
        sims = sorted(sims, key=lambda x: x[0], reverse=True)[: self.top_n_neighbors]
        num = sum(sim * rating for sim, rating in sims)
        den = sum(abs(sim) for sim, _ in sims)
        pred = num / den if den else self.global_mean
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(int(u), int(i)) for u, i in zip(pairs["user_id"], pairs["movie_id"])])

    def recommend(self, user_id: int, train_ratings: pd.DataFrame, k: int = 10, candidates: list[int] | None = None) -> pd.DataFrame:
        seen = set(train_ratings.loc[train_ratings["user_id"] == user_id, "movie_id"])
        if candidates is None:
            candidates = self.movie_ids
        scored = [(movie_id, self.predict_one(user_id, int(movie_id))) for movie_id in candidates if movie_id not in seen]
        return pd.DataFrame(scored, columns=["movie_id", "score"]).sort_values("score", ascending=False).head(k).reset_index(drop=True)

    def explain(self, user_id: int, movie_id: int, train_ratings: pd.DataFrame, top_n: int = 3) -> str:
        if self.similarity is None or movie_id not in self.similarity.index:
            return "Explanation unavailable because the movie was not present in training similarity matrix."
        user_seen = train_ratings[train_ratings["user_id"] == user_id][["movie_id", "rating", "title"]]
        contributions = []
        for _, row in user_seen.iterrows():
            seen_movie = int(row["movie_id"])
            if seen_movie in self.similarity.columns:
                sim = float(self.similarity.loc[movie_id, seen_movie])
                contributions.append((sim, row.get("title", f"Movie {seen_movie}"), float(row["rating"])))
        contributions = sorted(contributions, reverse=True)[:top_n]
        if not contributions:
            return "Recommended because its predicted rating is high for this user."
        parts = [f"{title} (rating {rating:.1f}, similarity {sim:.2f})" for sim, title, rating in contributions]
        return "Recommended because the user liked similar movies: " + "; ".join(parts)
