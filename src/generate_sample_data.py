from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MOVIES = [
    (1, 1999, "The Matrix"),
    (2, 1994, "The Shawshank Redemption"),
    (3, 2001, "Spirited Away"),
    (4, 2008, "The Dark Knight"),
    (5, 2010, "Inception"),
    (6, 1995, "Toy Story"),
    (7, 2003, "Finding Nemo"),
    (8, 2000, "Gladiator"),
    (9, 1997, "Titanic"),
    (10, 2002, "The Lord of the Rings: The Two Towers"),
    (11, 1993, "Jurassic Park"),
    (12, 2006, "The Departed"),
    (13, 2004, "Eternal Sunshine of the Spotless Mind"),
    (14, 1991, "The Silence of the Lambs"),
    (15, 2007, "Ratatouille"),
    (16, 2009, "Up"),
    (17, 1998, "Saving Private Ryan"),
    (18, 2005, "Batman Begins"),
]


def generate(path: str | Path = "data/sample", seed: int = 42):
    rng = np.random.default_rng(seed)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    n_users = 40
    n_movies = len(MOVIES)
    user_factors = rng.normal(size=(n_users, 4))
    item_factors = rng.normal(size=(n_movies, 4))
    user_bias = rng.normal(0, 0.35, size=n_users)
    item_bias = rng.normal(0, 0.25, size=n_movies)

    lines_by_movie = {movie_id: [] for movie_id, _, _ in MOVIES}
    for u in range(1, n_users + 1):
        # each user rates 6-13 movies; sparse enough for recommender behavior
        rated_items = rng.choice(np.arange(n_movies), size=rng.integers(6, 14), replace=False)
        start = pd.Timestamp("2003-01-01") + pd.Timedelta(days=int(rng.integers(0, 700)))
        for t, item_idx in enumerate(rated_items):
            raw = 3.45 + user_bias[u - 1] + item_bias[item_idx] + 0.65 * (user_factors[u - 1] @ item_factors[item_idx]) / 4
            rating = int(np.clip(np.round(raw + rng.normal(0, 0.55)), 1, 5))
            date = (start + pd.Timedelta(days=int(t * rng.integers(15, 70)))).date().isoformat()
            movie_id = MOVIES[item_idx][0]
            lines_by_movie[movie_id].append(f"{u},{rating},{date}")

    with (path / "combined_data_sample.txt").open("w", encoding="utf-8") as f:
        for movie_id, _, _ in MOVIES:
            f.write(f"{movie_id}:\n")
            for line in lines_by_movie[movie_id]:
                f.write(line + "\n")

    with (path / "movie_titles_sample.csv").open("w", encoding="latin1") as f:
        for movie_id, year, title in MOVIES:
            f.write(f"{movie_id},{year},{title}\n")


if __name__ == "__main__":
    generate()
