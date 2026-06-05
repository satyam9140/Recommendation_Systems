from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_dataset
from src.models.baseline import BiasBaselineRecommender
from src.preprocessing import filter_interactions, temporal_train_test_split


def test_sample_training_smoke():
    ratings, _ = load_dataset(ROOT / "data" / "sample")
    ratings = filter_interactions(ratings, min_user_ratings=3, min_movie_ratings=3)
    train, test = temporal_train_test_split(ratings)
    model = BiasBaselineRecommender().fit(train)
    preds = model.predict(test)
    assert len(preds) == len(test)
    assert preds.min() >= 1.0
    assert preds.max() <= 5.0
