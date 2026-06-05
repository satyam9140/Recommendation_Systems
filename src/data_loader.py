"""Netflix Prize data loading utilities.

Supports the official Kaggle Netflix Prize format:
- combined_data_1.txt ... combined_data_4.txt
- movie_titles.csv

Also supports the small sample dataset shipped in data/sample so the project
can be run instantly without downloading the 100M-row dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _combined_files(data_dir: Path) -> list[Path]:
    candidates = sorted(data_dir.glob("combined_data*.txt"))
    if not candidates:
        candidates = sorted(data_dir.glob("*.txt"))
    return candidates


def load_movie_titles(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    for name in ["movie_titles.csv", "movie_titles_sample.csv"]:
        path = data_dir / name
        if path.exists():
            # Netflix titles contain commas inside titles and are latin-1 encoded.
            rows = []
            with path.open("r", encoding="latin1") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    parts = line.split(",", 2)
                    if len(parts) == 3:
                        movie_id, year, title = parts
                        rows.append({
                            "movie_id": int(movie_id),
                            "year": None if year == "NULL" or year == "" else int(year),
                            "title": title,
                        })
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["movie_id", "year", "title"])


def load_ratings(
    data_dir: str | Path,
    max_movies: Optional[int] = None,
    max_ratings: Optional[int] = None,
    file_limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load Netflix combined_data files into a tidy DataFrame.

    Parameters
    ----------
    data_dir:
        Folder containing combined_data_*.txt files.
    max_movies:
        Optional cap on number of movie blocks to parse. Useful for quick experiments.
    max_ratings:
        Optional cap on rating rows to parse.
    file_limit:
        Optional cap on number of combined_data files.
    """
    data_dir = Path(data_dir)
    files = _combined_files(data_dir)
    if file_limit is not None:
        files = files[:file_limit]
    if not files:
        raise FileNotFoundError(
            f"No combined_data*.txt files found in {data_dir}. "
            "Use data/sample for the demo or place the Kaggle Netflix files in data/raw."
        )

    rows: list[dict] = []
    current_movie_id: Optional[int] = None
    movies_seen = 0

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                if line.endswith(":"):
                    current_movie_id = int(line[:-1])
                    movies_seen += 1
                    if max_movies is not None and movies_seen > max_movies:
                        return pd.DataFrame(rows)
                    continue
                if current_movie_id is None:
                    continue
                user_id, rating, date = line.split(",")
                rows.append({
                    "user_id": int(user_id),
                    "movie_id": int(current_movie_id),
                    "rating": float(rating),
                    "date": pd.to_datetime(date),
                })
                if max_ratings is not None and len(rows) >= max_ratings:
                    return pd.DataFrame(rows)

    return pd.DataFrame(rows)


def load_dataset(
    data_dir: str | Path,
    max_movies: Optional[int] = None,
    max_ratings: Optional[int] = None,
    file_limit: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = load_ratings(data_dir, max_movies=max_movies, max_ratings=max_ratings, file_limit=file_limit)
    titles = load_movie_titles(data_dir)
    if not titles.empty:
        ratings = ratings.merge(titles, on="movie_id", how="left")
    else:
        ratings["title"] = ratings["movie_id"].map(lambda x: f"Movie {x}")
        ratings["year"] = None
    return ratings, titles
