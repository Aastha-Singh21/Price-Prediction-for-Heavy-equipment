# Heavy Equipment Selling Price Prediction

Kaggle competition project predicting the resale price of heavy equipment 
using operational, transactional, and technical specification data.

## Overview
- 138,701 rows, 50 features (numeric + high-cardinality categorical)
- Evaluation metric: RMSLE (Root Mean Squared Log Error)
- Final Kaggle leaderboard score: 0.19544 RMSLE

## Approach
- Exploratory data analysis on target distribution, missing values, 
  and key feature relationships (equipment age, sale trends, categorical cardinality)
- Reusable preprocessing pipeline (label encoding, median imputation, 
  standard scaling) applied consistently to train/test
- Feature engineering: equipment age, sale year/month, and numeric 
  spec extraction from text fields
- Compared 3 models — Linear Regression (baseline), LightGBM, and XGBoost
- Hyperparameter tuning via manual iteration and RandomizedSearchCV
- Final submission: weighted ensemble of tuned LightGBM and XGBoost

## Tech stack
Python, pandas, NumPy, scikit-learn, LightGBM, XGBoost, matplotlib, seaborn
