# Recommendation Systems for Personalized Content Discovery

A complete, reproducible machine-learning project for the **Netflix Prize Dataset** recommendation-system problem statement.

The project implements:

- Exploratory Data Analysis (EDA)
- Data processing pipeline for the official Netflix Prize `combined_data_*.txt` format
- Two-plus recommendation models:
  - Bias baseline recommender
  - Item-based collaborative filtering
  - Matrix factorization with SGD
- Rating prediction evaluation using RMSE and MAE
- Ranking evaluation using MAP@10, Precision@10, Recall@10, and HitRate@10
- Top-K personalized recommendations
- Explainable recommendation examples
- Streamlit dashboard
- Technical report PDF and 8-slide presentation PDF

> The full Netflix Prize dataset contains about 100M ratings, so it is not included in this zip. A small demo dataset is included under `data/sample/` so the whole project runs immediately. For final competition results, download the official Kaggle dataset and place it under `data/raw/`.

---

## 1. Project Structure

```text
netflix_recommender_project/
├── app/
│   └── streamlit_app.py
├── data/
│   └── sample/
│       ├── combined_data_sample.txt
│       └── movie_titles_sample.csv
├── notebooks/
│   └── 01_end_to_end_demo.ipynb
├── outputs/
│   └── demo/
│       ├── metrics.csv
│       ├── eda_summary.json
│       ├── figures/
│       └── recommendations/
├── presentation/
│   └── recommendation_presentation.pdf
├── reports/
│   └── technical_report.pdf
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── recommend.py
│   ├── train.py
│   └── models/
│       ├── baseline.py
│       ├── item_cf.py
│       └── matrix_factorization.py
├── tests/
│   └── test_smoke.py
├── requirements.txt
└── README.md
```

---

## 2. Setup

### Local setup

```bash
cd netflix_recommender_project
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Google Colab setup

Upload the zip, unzip it, then run:

```python
%cd netflix_recommender_project
!pip install -r requirements.txt
```

---

## 3. Run the Demo Project

The demo uses the included sample data.

```bash
python -m src.train --data-dir data/sample --output-dir outputs/demo
```

Expected outputs:

- `outputs/demo/metrics.csv`
- `outputs/demo/eda_summary.json`
- `outputs/demo/figures/*.png`
- `outputs/demo/recommendations/*_sample_recommendations.csv`
- `outputs/demo/models/*.joblib`

---

## 4. Run the Dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard features:

- Select a user
- View Top-K recommendations
- View recommendation scores
- View user watch/rating history
- See explanation for item-based recommendations
- Compare model metrics

---

## 5. Use the Real Netflix Prize Dataset

Download from Kaggle:

`https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data`

Put the files here:

```text
data/raw/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
└── movie_titles.csv
```

Then run a manageable subset first:

```bash
python -m src.train \
  --data-dir data/raw \
  --output-dir outputs/netflix_subset \
  --max-movies 1000 \
  --max-ratings 300000 \
  --min-user-ratings 5 \
  --min-movie-ratings 10 \
  --map-max-users 1000
```

For larger runs, increase `--max-movies`, `--max-ratings`, and `--map-max-users` gradually. Do not jump directly to 100M ratings unless you have enough RAM and time.

---

## 6. Evaluation Methodology

### Train-test split

The project uses a **user-wise temporal split**:

1. Sort each user's ratings by date.
2. Use older interactions for training.
3. Hold out the latest interactions for testing.

This is stricter than random splitting because it imitates production behavior: learn from the past and recommend future items.

### RMSE

RMSE measures how close predicted ratings are to actual ratings. Lower is better.

### MAP@10

For MAP@10, a movie is considered relevant if:

```text
actual rating >= 3.5
```

For each evaluated user:

1. Generate Top-10 unseen movie recommendations.
2. Compare the ranked list against that user's relevant held-out test movies.
3. Compute average precision.
4. Average across users.

MAP@10 is more aligned with recommendation quality than RMSE because it rewards putting relevant movies near the top.

---

## 7. Models Implemented

### Bias Baseline

Predicts rating using:

```text
global mean + user bias + item bias
```

Use this as the sanity-check model. If a complex model cannot beat it, that complex model is not worth presenting.

### Item-Based Collaborative Filtering

Computes item similarity from user rating patterns and recommends movies similar to what a user liked before.

Strengths:

- Explainable
- Practical
- Good for dashboards

Weaknesses:

- Struggles with cold-start movies
- Similarity matrix can become large

### Matrix Factorization SGD

Learns latent user and movie vectors to capture hidden preference patterns.

Strengths:

- Strong rating prediction baseline
- Captures latent taste dimensions

Weaknesses:

- Less explainable
- Needs tuning
- Can overfit small/sparse data

---

## 8. Competition-Level Improvements

For a stronger final submission:

1. Train on a larger Netflix subset.
2. Add time-aware features, such as rating recency.
3. Add model ensembling: bias baseline + item-CF + matrix factorization.
4. Tune matrix factorization hyperparameters.
5. Add diversity re-ranking so recommendations are not all from the same genre/style.
6. Add cold-start logic using popularity and metadata.

---

## 9. Brutal Mentor Notes

A weak project only prints RMSE and calls it a recommender. That is not enough.

Your submission becomes defensible only when you can answer:

- Why does MAP@10 matter more than RMSE for content discovery?
- What happens for new users and new movies?
- Why did you choose a temporal split instead of a random split?
- Can the model explain why a movie was recommended?
- Does the dashboard prove business usability or just show tables?

This package is structured to answer those questions cleanly.
