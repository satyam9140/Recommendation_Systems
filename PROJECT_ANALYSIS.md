# Problem Statement Analysis

The uploaded document asks teams to build a recommendation system for personalized content discovery using the Netflix Prize dataset.

## What the judges will actually care about

1. **EDA is not optional decoration.** It must explain sparsity, user activity imbalance, movie popularity concentration, and rating distribution.
2. **At least two models are required.** A single SVD model is not enough because the statement explicitly asks for comparison.
3. **RMSE alone is a trap.** RMSE measures rating prediction, but a recommender lives or dies by whether relevant items appear in the top positions.
4. **MAP@10 must be implemented correctly.** Relevance is defined as actual rating >= 3.5.
5. **Top-K recommendations must be demonstrated.** The deliverable should show sample users, recommended movies, scores, success cases, and failure cases.
6. **Reproducibility matters.** The repo needs clear run commands, dependency list, and sensible handling of the huge original dataset.

## Chosen solution strategy

This project implements three models:

- Bias baseline
- Item-based collaborative filtering
- Matrix factorization with SGD

The comparison covers:

- Recommendation quality
- Training complexity
- Computational efficiency
- Practical usability

## Why this approach is defensible

The solution is deliberately not overcomplicated. It uses models that can be explained, evaluated, and reproduced. The included dashboard makes the system usable, while the code supports scaling from demo data to the real Netflix Prize files.
