from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_loader import load_dataset
from src.models.item_cf import ItemItemCFRecommender
from src.preprocessing import filter_interactions
from src.recommend import attach_titles

st.set_page_config(page_title="Netflix Recommendation Engine", layout="wide")
st.title("Personalized Netflix Recommendation System")
st.caption("Demo dashboard for Top-K movie recommendations, similar content, and recommendation explanations.")

output_dir = ROOT / "outputs" / "demo"
data_dir = ROOT / "data" / "sample"
model_path = output_dir / "models" / "best_model.joblib"

@st.cache_data
def load_data():
    ratings, titles = load_dataset(data_dir)
    ratings = filter_interactions(ratings, min_user_ratings=3, min_movie_ratings=3)
    return ratings, titles

@st.cache_resource
def load_model(ratings):
    if model_path.exists():
        return joblib.load(model_path)
    # fallback: train quickly if user opened dashboard before running train.py
    model = ItemItemCFRecommender(top_n_neighbors=30)
    model.fit(ratings)
    return model

ratings, titles = load_data()
model = load_model(ratings)

left, right = st.columns([1, 2])
with left:
    st.subheader("User Controls")
    users = sorted(ratings["user_id"].unique().tolist())
    user_id = st.selectbox("Select user", users)
    k = st.slider("Number of recommendations", 5, 15, 10)

    st.metric("Ratings", len(ratings))
    st.metric("Users", ratings["user_id"].nunique())
    st.metric("Movies", ratings["movie_id"].nunique())

with right:
    st.subheader("Top Recommendations")
    recs = model.recommend(int(user_id), ratings, k=k, candidates=sorted(ratings["movie_id"].unique().tolist()))
    recs = attach_titles(recs, titles)
    st.dataframe(recs[["movie_id", "title", "score"]], use_container_width=True)

st.divider()
col1, col2 = st.columns(2)
with col1:
    st.subheader("User History")
    history = ratings[ratings["user_id"] == user_id].sort_values("date", ascending=False)
    st.dataframe(history[["date", "movie_id", "title", "rating"]], use_container_width=True)

with col2:
    st.subheader("Why This Recommendation?")
    if not recs.empty:
        movie_id = int(st.selectbox("Choose a recommended movie", recs["movie_id"].tolist(), format_func=lambda x: titles.set_index("movie_id").loc[x, "title"] if x in titles["movie_id"].values else str(x)))
        if hasattr(model, "explain"):
            st.write(model.explain(int(user_id), movie_id, ratings))
        else:
            st.write("This model recommends the movie because its predicted score is high for the selected user.")

st.divider()
st.subheader("Model Comparison")
metrics_path = output_dir / "metrics.csv"
if metrics_path.exists():
    st.dataframe(pd.read_csv(metrics_path), use_container_width=True)
else:
    st.info("Run `python -m src.train` from the project root to generate metrics.csv.")
