# 🎬 Recommendation Systems for Personalized Content Discovery

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?style=for-the-badge&logo=streamlit)
![Status](https://img.shields.io/badge/Project-Completed-success?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-Recommendation%20Systems-orange?style=for-the-badge)

### Personalized movie recommendation engine using collaborative filtering, matrix factorization, and ranking-based evaluation

</div>

---

# 📌 Project Overview

This project builds a complete **Recommendation System** for personalized content discovery using the Netflix Prize style rating dataset.

The system is designed to recommend movies to users based on historical rating behavior. It includes a full machine-learning workflow from data loading and preprocessing to model training, evaluation, recommendation generation, explainability, reporting, and dashboard visualization.

The project implements:

- 📊 Exploratory Data Analysis
- 🧹 Data preprocessing pipeline
- 🤖 Multiple recommendation models
- 📈 Rating prediction and ranking evaluation
- 🎯 Top-K personalized recommendations
- 🧠 Explainable recommendation examples
- 📺 Streamlit dashboard
- 📄 Technical report and presentation material

---

# 🎯 Problem Statement

Modern streaming platforms have thousands of movies, shows, and content items. Users cannot manually explore everything, so platforms need intelligent recommendation systems that can predict what a user is likely to prefer.

A weak recommendation project only predicts ratings and stops there. That is not enough.

A practical recommender must answer:

- Which movies should be recommended to a user?
- How accurate are the predicted ratings?
- Are relevant movies appearing in the Top-K list?
- Can the system explain why an item was recommended?
- Can the system work on both sample data and larger real-world datasets?

This project solves the problem by building and comparing multiple recommender models using both **rating prediction metrics** and **ranking-based recommendation metrics**.

---

# 🧠 System Architecture

```text
┌──────────────────────────────────────┐
│        Netflix Rating Dataset        │
└──────────────────┬───────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Data Loading       │
        │   & Parsing          │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │   Preprocessing      │
        │   & Train-Test Split │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Exploratory Data     │
        │ Analysis             │
        └──────────┬───────────┘
                   │
                   ▼
┌──────────────────┼──────────────────┐
│                  │                  │
▼                  ▼                  ▼
Bias Baseline   Item-Based CF   Matrix Factorization
Recommender     Recommender     Recommender
│                  │                  │
└──────────────────┼──────────────────┘
                   ▼
        ┌──────────────────────┐
        │ Model Evaluation     │
        │ RMSE, MAE, MAP@10    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Top-K Personalized   │
        │ Recommendations      │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Streamlit Dashboard  │
        │ & Reports            │
        └──────────────────────┘
```

---

# 📂 Project Structure

```text
Recommendation_Systems/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── sample/
│       ├── combined_data_sample.txt
│       └── movie_titles_sample.csv
│
├── notebooks/
│   └── 01_end_to_end_demo.ipynb
│
├── outputs/
│   └── demo/
│       ├── metrics.csv
│       ├── eda_summary.json
│       ├── figures/
│       ├── recommendations/
│       └── models/
│
├── presentation/
│   └── recommendation_presentation.pdf
│
├── reports/
│   └── technical_report.pdf
│
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
│
├── tests/
│   └── test_smoke.py
│
├── PROJECT_ANALYSIS.md
├── requirements.txt
├── run_demo.bat
├── run_demo.sh
└── README.md
```

---

# 📊 Dataset Used

The project is built around the **Netflix Prize Dataset** format.

The full Netflix Prize dataset is very large and contains around 100 million ratings, so it is not included directly in the repository.

A small sample dataset is included inside:

```text
data/sample/
```

This allows the project to run immediately for demonstration and testing.

For full-scale experimentation, download the Netflix Prize dataset from Kaggle and place the raw files inside:

```text
data/raw/
├── combined_data_1.txt
├── combined_data_2.txt
├── combined_data_3.txt
├── combined_data_4.txt
└── movie_titles.csv
```

Dataset source:

```text
https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
```

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Streamlit
- Jupyter Notebook
- Recommendation Systems
- Collaborative Filtering
- Matrix Factorization

---

# 🚀 Workflow

## 1️⃣ Data Loading

The project reads Netflix-style rating files and movie metadata.

Main responsibilities:

- Parse combined rating files
- Load movie titles
- Convert raw ratings into structured tabular format
- Prepare user-item interaction data

---

## 2️⃣ Data Preprocessing

The preprocessing pipeline prepares the dataset for model training.

Key steps include:

- Cleaning rating records
- Filtering users and movies
- Handling sparse interaction data
- Creating train-test splits
- Preparing user-item mappings

The project uses a **temporal train-test split**, which is more realistic than random splitting because the model learns from older interactions and is tested on future interactions.

---

## 3️⃣ Exploratory Data Analysis

EDA helps understand the behavior of the recommendation dataset.

The project analyzes:

- Rating distribution
- User activity levels
- Movie popularity
- Sparsity of user-item interactions
- Dataset summary statistics

Generated EDA outputs are saved inside:

```text
outputs/demo/figures/
outputs/demo/eda_summary.json
```

---

# 🤖 Models Implemented

## 1. Bias Baseline Recommender

This is the sanity-check model.

It predicts ratings using:

```text
Global average rating + User bias + Movie bias
```

This model is simple but important. If a complex model cannot beat this baseline, the complex model is probably not useful.

---

## 2. Item-Based Collaborative Filtering

This model recommends movies similar to the ones a user has liked before.

It uses user rating patterns to calculate similarity between movies.

Strengths:

- Easy to explain
- Useful for dashboards
- Works well when item similarity is meaningful

Weaknesses:

- Struggles with new movies
- Similarity matrix can become large
- Performance depends heavily on rating density

---

## 3. Matrix Factorization with SGD

This model learns hidden user and movie vectors.

It tries to capture latent taste patterns such as:

- Genre preference
- Popularity preference
- Rating behavior
- Similarity in hidden preference space

Strengths:

- Strong recommendation baseline
- Captures hidden user-item relationships
- Better for sparse data than simple similarity methods

Weaknesses:

- Less explainable than item-based filtering
- Requires tuning
- Can overfit on small datasets

---

# 📈 Evaluation Metrics

The project evaluates recommendation quality using both rating prediction and ranking metrics.

| Metric | Purpose |
|---|---|
| RMSE | Measures rating prediction error |
| MAE | Measures average absolute rating error |
| MAP@10 | Measures ranking quality of top 10 recommendations |
| Precision@10 | Measures how many recommended items are relevant |
| Recall@10 | Measures how many relevant items were retrieved |
| HitRate@10 | Measures whether at least one relevant item appears in Top 10 |

## Why ranking metrics matter

RMSE and MAE only tell whether the predicted ratings are close to actual ratings.

But a recommendation system is not built only to predict ratings. It is built to show the right items at the top.

That is why MAP@10, Precision@10, Recall@10, and HitRate@10 are more useful for judging real recommendation quality.

---

# 📊 Key Outputs

After running the project, outputs are generated inside:

```text
outputs/demo/
```

Important outputs include:

- `metrics.csv`
- `eda_summary.json`
- EDA visualizations
- Trained model files
- Personalized recommendation CSV files
- Model comparison results

---

# 📺 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

Run it using:

```bash
streamlit run app/streamlit_app.py
```

Dashboard features:

- Select a user
- View personalized Top-K recommendations
- Compare model metrics
- View recommendation scores
- Explore user rating history
- See explainable recommendation examples

---

# ▶️ How to Run the Project

## Step 1 — Clone the Repository

```bash
git clone https://github.com/satyam9140/Recommendation_Systems.git
cd Recommendation_Systems
```

---

## Step 2 — Create Virtual Environment

```bash
python -m venv .venv
```

---

## Step 3 — Activate Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 5 — Run Demo Training Pipeline

```bash
python -m src.train --data-dir data/sample --output-dir outputs/demo
```

Expected outputs:

```text
outputs/demo/metrics.csv
outputs/demo/eda_summary.json
outputs/demo/figures/
outputs/demo/recommendations/
outputs/demo/models/
```

---

## Step 6 — Run Dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# 🧪 Run Tests

To verify that the basic project pipeline is working:

```bash
pytest tests/
```

If `pytest` is not installed:

```bash
pip install pytest
pytest tests/
```

---

# 📌 Project Highlights

✅ End-to-end recommendation-system pipeline

✅ Multiple recommender models

✅ Rating prediction and ranking evaluation

✅ Personalized Top-K recommendation generation

✅ Explainable recommendation examples

✅ Streamlit dashboard integration

✅ Sample dataset included for immediate execution

✅ Scalable structure for larger Netflix Prize dataset

✅ Technical report and presentation included

---

# 🔮 Future Improvements

- Add user-based collaborative filtering
- Add content-based filtering using movie metadata
- Add hybrid recommender system
- Add deep learning recommenders
- Add implicit feedback modeling
- Add genre diversity and novelty metrics
- Add cold-start handling for new users and new movies
- Add deployment support using Streamlit Cloud or Vercel frontend
- Add automated hyperparameter tuning
- Add model monitoring for recommendation drift

---

# 🧠 Brutal Mentor Notes

A recommendation project becomes weak when it only says:

```text
"I built a model and got RMSE."
```

That is not enough.

A defensible recommendation system must prove:

- Why the recommended movies are relevant
- Whether the Top-K list is actually useful
- Whether the model beats a simple baseline
- How it handles sparse users and unpopular movies
- How it explains recommendations
- Whether the dashboard supports real decision-making

This project is structured to answer those questions instead of only showing a model score.

---

# 🏁 Conclusion

This project demonstrates how recommendation systems can personalize content discovery using collaborative filtering, baseline models, and matrix factorization.

The system provides:

- Personalized movie recommendations
- Model comparison using multiple metrics
- Explainable recommendation outputs
- Practical dashboard-based visualization
- A reproducible machine-learning pipeline

It is a strong academic and portfolio project because it covers not only model training, but also evaluation, explainability, reporting, and dashboard presentation.

---

# 👨‍💻 Author

**Satyam Singh**

Domain: Data Science & AI  
Project: Recommendation Systems for Personalized Content Discovery  
Repository: `Recommendation_Systems`
