from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.data_loader import load_dataset
from src.eda import compute_eda_summary, save_eda_figures
from src.evaluation import map_at_k, rating_metrics
from src.models.baseline import BiasBaselineRecommender
from src.models.item_cf import ItemItemCFRecommender
from src.models.matrix_factorization import MatrixFactorizationSGD
from src.preprocessing import filter_interactions, temporal_train_test_split
from src.recommend import generate_recommendations_for_users


def build_models(random_state: int = 42) -> list:
    return [
        BiasBaselineRecommender(shrinkage=10),
        ItemItemCFRecommender(top_n_neighbors=30),
        MatrixFactorizationSGD(n_factors=24, n_epochs=35, learning_rate=0.02, reg=0.04, random_state=random_state),
    ]


def run_pipeline(args: argparse.Namespace) -> dict:
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    model_dir = output_dir / "models"
    rec_dir = output_dir / "recommendations"
    for d in [output_dir, figure_dir, model_dir, rec_dir]:
        d.mkdir(parents=True, exist_ok=True)

    ratings, titles = load_dataset(args.data_dir, max_movies=args.max_movies, max_ratings=args.max_ratings, file_limit=args.file_limit)
    ratings = filter_interactions(ratings, min_user_ratings=args.min_user_ratings, min_movie_ratings=args.min_movie_ratings)
    if ratings.empty:
        raise ValueError("No ratings left after filtering. Lower min_user_ratings/min_movie_ratings.")

    eda_summary = compute_eda_summary(ratings)
    save_eda_figures(ratings, figure_dir)
    (output_dir / "eda_summary.json").write_text(json.dumps(eda_summary, indent=2), encoding="utf-8")

    train_df, test_df = temporal_train_test_split(ratings, test_ratio=args.test_ratio, min_train_items=2)
    train_df.to_csv(output_dir / "train_split.csv", index=False)
    test_df.to_csv(output_dir / "test_split.csv", index=False)

    all_movie_ids = sorted(ratings["movie_id"].unique().tolist())
    metrics_rows = []
    sample_users = sorted(test_df["user_id"].unique().tolist())[: args.n_sample_users]

    best_model = None
    best_map = -1.0
    for model in build_models(random_state=args.random_state):
        print(f"Training {model.name}...")
        model.fit(train_df)
        pred = model.predict(test_df)
        row = {"model": model.name, **rating_metrics(test_df["rating"].to_numpy(), pred)}
        rank_metrics = map_at_k(
            model,
            train_df=train_df,
            test_df=test_df,
            all_movie_ids=all_movie_ids,
            k=args.top_k,
            relevant_threshold=args.relevant_threshold,
            max_users=args.map_max_users,
        )
        row.update(rank_metrics)
        metrics_rows.append(row)
        joblib.dump(model, model_dir / f"{model.name}.joblib")

        recs = generate_recommendations_for_users(model, sample_users, train_df, titles, k=args.top_k)
        recs.to_csv(rec_dir / f"{model.name}_sample_recommendations.csv", index=False)
        if row["MAP@10"] > best_map:
            best_map = row["MAP@10"]
            best_model = model

    metrics = pd.DataFrame(metrics_rows).sort_values(["MAP@10", "RMSE"], ascending=[False, True])
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    if best_model is not None:
        joblib.dump(best_model, model_dir / "best_model.joblib")

    print("\nModel comparison:")
    print(metrics.to_string(index=False))
    return {"eda": eda_summary, "metrics": metrics_rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Netflix recommendation models.")
    parser.add_argument("--data-dir", default="data/sample", help="Folder with combined_data*.txt and movie_titles.csv")
    parser.add_argument("--output-dir", default="outputs/demo")
    parser.add_argument("--max-movies", type=int, default=None, help="Optional movie block cap for real Netflix data")
    parser.add_argument("--max-ratings", type=int, default=None, help="Optional rating row cap for real Netflix data")
    parser.add_argument("--file-limit", type=int, default=None, help="Optional combined file cap")
    parser.add_argument("--min-user-ratings", type=int, default=3)
    parser.add_argument("--min-movie-ratings", type=int, default=3)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--relevant-threshold", type=float, default=3.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--map-max-users", type=int, default=None, help="Use a cap for faster MAP on large data")
    parser.add_argument("--n-sample-users", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
