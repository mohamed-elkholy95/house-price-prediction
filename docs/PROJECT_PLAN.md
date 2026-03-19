# Project Plan: House Price Prediction

> **Status: COMPLETE**  
> Full implementation at: `/home/mac/projects/house-price-prediction/`

This document summarizes the completed project. The actual code, notebooks, and artifacts live in the dedicated project directory.

---

## Problem Statement

Predict residential property sale prices in Ames, Iowa using 79 explanatory variables describing almost every aspect of the home. This is a regression problem requiring robust feature engineering and ensemble modeling.

**Business Value:** Accurate price estimates help buyers, sellers, and realtors make data-driven decisions. Production-grade models can power automated valuation systems (AVMs) used by mortgage lenders and iBuyers.

## Dataset

- **Source:** Kaggle — House Prices: Advanced Regression Techniques
- **Size:** 1,460 training samples, 1,459 test samples, 79 features
- **Target:** `SalePrice` (continuous, right-skewed — log-transformed for modeling)
- **Feature Types:** 43 categorical, 36 numeric

### Feature Categories
- **Location:** Neighborhood, Condition1/2, Zoning
- **Structure:** YearBuilt, YearRemodAdd, BldgType, HouseStyle
- **Space:** GrLivArea, TotalBsmtSF, GarageArea, PoolArea
- **Quality:** OverallQual, OverallCond, ExterQual, KitchenQual
- **Utilities:** Heating, CentralAir, Electrical

## Implementation Summary

### Phase 1: EDA ✅
- Distribution of SalePrice (right-skewed → log transform)
- Correlation analysis (GrLivArea, OverallQual strongest predictors)
- Outlier detection and removal (GrLivArea > 4000 with low prices)
- Missing value heatmap (60+ features with NAs)

### Phase 2: Feature Engineering ✅
- **Missing Data:** KNN imputation (numeric), mode (categorical), domain-specific (e.g., LotFrontage by Neighborhood median)
- **Log Transforms:** SalePrice, GrLivArea, LotArea, TotalBsmtSF
- **New Features:**
  - `TotalSF` = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
  - `TotalBathrooms` = full + 0.5×half baths (basement + above)
  - `HouseAge` = YrSold - YearBuilt
  - `RemodAge` = YrSold - YearRemodAdd
  - `HasPool`, `Has2ndFloor`, `HasGarage`, `HasBsmt` (binary flags)
  - Polynomial interactions for top numeric features
- **Encoding:** Label encoding (ordinal qualities), one-hot (nominal)
- **Scaling:** RobustScaler (outlier-resistant)

### Phase 3: Models ✅

| Model | Notes |
|---|---|
| Random Forest | 500 trees, max_features='sqrt', oob_score=True |
| XGBoost | 1000 rounds, lr=0.05, max_depth=3, early stopping |
| LightGBM | dart boosting, feature fraction=0.8 |
| Ridge/Lasso | Regularized linear baselines |
| Stacking | RF + XGB + LGBM → Ridge meta-learner |

### Phase 4: Interpretability ✅
- SHAP TreeExplainer for gradient boosting models
- Global: beeswarm, bar, heatmap plots
- Local: waterfall, force plots for individual predictions
- Partial dependence plots (OverallQual, GrLivArea, Neighborhood)

### Phase 5: Streamlit Dashboard ✅
- Input form: key property characteristics
- Live price prediction with confidence interval
- SHAP waterfall showing top contributing factors
- Neighborhood comparison chart
- Model performance comparison

### Phase 6: Testing ✅
- Data pipeline unit tests
- Feature engineering tests (expected column count, no NaN post-imputation)
- Model loading and prediction smoke tests
- Streamlit app integration test

### Phase 7: Containerization ✅
- Multi-stage Dockerfile (build + runtime)
- `docker-compose.yml` for local development
- Model artifacts baked into image

## File Reference

```
/home/mac/projects/house-price-prediction/
├── data/
│   ├── raw/train.csv, test.csv
│   └── processed/
├── src/
│   ├── data/loader.py, preprocessor.py
│   ├── features/engineer.py
│   ├── models/rf.py, xgb.py, lgbm.py, stacking.py
│   └── evaluation/metrics.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling_and_evaluation.ipynb
├── streamlit_app/app.py
├── Dockerfile
└── tests/
```

## Tech Stack

`Python 3.11` `scikit-learn` `XGBoost` `LightGBM` `SHAP` `Streamlit` `Plotly` `pandas` `numpy` `Docker` `pytest`
