from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rating_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
    }


def average_precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return np.nan
    hits = 0
    score = 0.0
    for idx, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            hits += 1
            score += hits / idx
    return score / min(len(relevant), k)


def precision_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if k == 0:
        return 0.0
    return len(set(recommended[:k]) & relevant) / k


def recall_at_k(recommended: list[int], relevant: set[int], k: int = 10) -> float:
    if not relevant:
        return np.nan
    return len(set(recommended[:k]) & relevant) / len(relevant)


def map_at_k(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_movie_ids: list[int],
    k: int = 10,
    relevant_threshold: float = 3.5,
    max_users: int | None = None,
) -> dict[str, float]:
    ap_scores = []
    precision_scores = []
    recall_scores = []
    hit_scores = []
    users = test_df["user_id"].unique().tolist()
    if max_users is not None:
        users = users[:max_users]
    for user_id in users:
        user_test = test_df[test_df["user_id"] == user_id]
        relevant = set(user_test.loc[user_test["rating"] >= relevant_threshold, "movie_id"].astype(int))
        if not relevant:
            continue
        recs = model.recommend(int(user_id), train_df, k=k, candidates=all_movie_ids)
        recommended = recs["movie_id"].astype(int).tolist()
        ap_scores.append(average_precision_at_k(recommended, relevant, k=k))
        precision_scores.append(precision_at_k(recommended, relevant, k=k))
        recall_scores.append(recall_at_k(recommended, relevant, k=k))
        hit_scores.append(float(len(set(recommended) & relevant) > 0))
    if not ap_scores:
        return {"MAP@10": 0.0, "Precision@10": 0.0, "Recall@10": 0.0, "HitRate@10": 0.0, "EvaluatedUsers": 0}
    return {
        "MAP@10": float(np.nanmean(ap_scores)),
        "Precision@10": float(np.nanmean(precision_scores)),
        "Recall@10": float(np.nanmean(recall_scores)),
        "HitRate@10": float(np.nanmean(hit_scores)),
        "EvaluatedUsers": int(len(ap_scores)),
    }


def catalog_coverage(recommendations: pd.DataFrame, all_movie_ids: list[int]) -> float:
    if len(all_movie_ids) == 0 or recommendations.empty:
        return 0.0
    return recommendations["movie_id"].nunique() / len(set(all_movie_ids))
