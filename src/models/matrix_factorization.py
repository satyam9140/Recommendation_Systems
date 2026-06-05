from __future__ import annotations

import numpy as np
import pandas as pd


class MatrixFactorizationSGD:
    """Explicit-feedback matrix factorization trained with SGD.

    This is a lightweight SVD-style latent factor model with user and item bias.
    It is intentionally dependency-light so it runs in Colab without Surprise.
    """

    name = "matrix_factorization_sgd"

    def __init__(
        self,
        n_factors: int = 24,
        n_epochs: int = 25,
        learning_rate: float = 0.015,
        reg: float = 0.05,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.reg = reg
        self.random_state = random_state
        self.global_mean = 3.0
        self.user_to_idx: dict[int, int] = {}
        self.item_to_idx: dict[int, int] = {}
        self.idx_to_item: list[int] = []
        self.P: np.ndarray | None = None
        self.Q: np.ndarray | None = None
        self.bu: np.ndarray | None = None
        self.bi: np.ndarray | None = None
        self.movie_ids: list[int] = []

    def fit(self, ratings: pd.DataFrame):
        rng = np.random.default_rng(self.random_state)
        users = sorted(ratings["user_id"].unique().tolist())
        items = sorted(ratings["movie_id"].unique().tolist())
        self.user_to_idx = {int(u): idx for idx, u in enumerate(users)}
        self.item_to_idx = {int(i): idx for idx, i in enumerate(items)}
        self.idx_to_item = [int(i) for i in items]
        self.movie_ids = self.idx_to_item.copy()
        self.global_mean = float(ratings["rating"].mean())
        n_users, n_items = len(users), len(items)
        self.P = 0.1 * rng.normal(size=(n_users, self.n_factors))
        self.Q = 0.1 * rng.normal(size=(n_items, self.n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)

        train_array = ratings[["user_id", "movie_id", "rating"]].to_numpy()
        for _ in range(self.n_epochs):
            rng.shuffle(train_array)
            for user_id, movie_id, rating in train_array:
                u = self.user_to_idx.get(int(user_id))
                i = self.item_to_idx.get(int(movie_id))
                if u is None or i is None:
                    continue
                pred = self.global_mean + self.bu[u] + self.bi[i] + float(self.P[u] @ self.Q[i])
                err = float(rating) - pred
                self.bu[u] += self.learning_rate * (err - self.reg * self.bu[u])
                self.bi[i] += self.learning_rate * (err - self.reg * self.bi[i])
                pu = self.P[u].copy()
                qi = self.Q[i].copy()
                self.P[u] += self.learning_rate * (err * qi - self.reg * pu)
                self.Q[i] += self.learning_rate * (err * pu - self.reg * qi)
        return self

    def predict_one(self, user_id: int, movie_id: int) -> float:
        if self.P is None or self.Q is None or self.bu is None or self.bi is None:
            raise RuntimeError("Model is not fitted.")
        u = self.user_to_idx.get(int(user_id))
        i = self.item_to_idx.get(int(movie_id))
        pred = self.global_mean
        if u is not None:
            pred += self.bu[u]
        if i is not None:
            pred += self.bi[i]
        if u is not None and i is not None:
            pred += float(self.P[u] @ self.Q[i])
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, pairs: pd.DataFrame) -> np.ndarray:
        return np.array([self.predict_one(int(u), int(i)) for u, i in zip(pairs["user_id"], pairs["movie_id"])])

    def recommend(self, user_id: int, train_ratings: pd.DataFrame, k: int = 10, candidates: list[int] | None = None) -> pd.DataFrame:
        seen = set(train_ratings.loc[train_ratings["user_id"] == user_id, "movie_id"])
        if candidates is None:
            candidates = self.movie_ids
        scored = [(movie_id, self.predict_one(user_id, int(movie_id))) for movie_id in candidates if movie_id not in seen]
        return pd.DataFrame(scored, columns=["movie_id", "score"]).sort_values("score", ascending=False).head(k).reset_index(drop=True)
