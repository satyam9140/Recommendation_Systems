from __future__ import annotations

import numpy as np
import pandas as pd


class BiasBaselineRecommender:
    """Global mean + user bias + item bias baseline.

    This simple model is important because many complex recommenders fail to beat
    it on sparse data. It also gives a clean benchmark for RMSE.
    """

    name = "bias_baseline"

    def __init__(self, shrinkage: float = 10.0):
        self.shrinkage = shrinkage
        self.global_mean = 3.0
        self.user_bias: dict[int, float] = {}
        self.item_bias: dict[int, float] = {}
        self.movie_ids: list[int] = []
        self.popularity: pd.Series | None = None

    def fit(self, ratings: pd.DataFrame):
        self.global_mean = float(ratings["rating"].mean())
        self.movie_ids = sorted(ratings["movie_id"].unique().tolist())
        user_stats = ratings.groupby("user_id")["rating"].agg(["mean", "count"])
        self.user_bias = ((user_stats["mean"] - self.global_mean) * user_stats["count"] / (user_stats["count"] + self.shrinkage)).to_dict()

        tmp = ratings.copy()
        tmp["user_bias"] = tmp["user_id"].map(self.user_bias).fillna(0.0)
        tmp["residual"] = tmp["rating"] - self.global_mean - tmp["user_bias"]
        item_stats = tmp.groupby("movie_id")["residual"].agg(["mean", "count"])
        self.item_bias = (item_stats["mean"] * item_stats["count"] / (item_stats["count"] + self.shrinkage)).to_dict()
        self.popularity = ratings.groupby("movie_id")["rating"].agg(["count", "mean"]).sort_values(["count", "mean"], ascending=False)["mean"]
        return self

    def predict_one(self, user_id: int, movie_id: int) -> float:
        pred = self.global_mean + self.user_bias.get(user_id, 0.0) + self.item_bias.get(movie_id, 0.0)
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(int(u), int(i)) for u, i in zip(pairs["user_id"], pairs["movie_id"])])

    def recommend(self, user_id: int, train_ratings: pd.DataFrame, k: int = 10, candidates: list[int] | None = None) -> pd.DataFrame:
        seen = set(train_ratings.loc[train_ratings["user_id"] == user_id, "movie_id"])
        if candidates is None:
            candidates = self.movie_ids
        scored = []
        for movie_id in candidates:
            if movie_id in seen:
                continue
            scored.append((movie_id, self.predict_one(user_id, movie_id)))
        recs = pd.DataFrame(scored, columns=["movie_id", "score"]).sort_values("score", ascending=False).head(k)
        return recs.reset_index(drop=True)
