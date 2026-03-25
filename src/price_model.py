"""
House Price Prediction Model
=============================

This module implements a complete regression pipeline for predicting house prices.
It demonstrates three classical ML approaches:

1. **Linear Regression** — Baseline model assuming a linear relationship between
   features and price. Fast to train, interpretable coefficients, but may underfit
   non-linear patterns.

2. **Random Forest** — Ensemble of decision trees trained on bootstrap samples.
   Reduces variance through averaging (bagging), handles non-linearity well,
   and provides built-in feature importance rankings.

3. **Gradient Boosting** — Sequential ensemble where each tree corrects the
   residual errors of the previous one. Typically achieves the best accuracy
   but is more sensitive to hyperparameters and prone to overfitting.

Key concepts demonstrated:
- Synthetic data generation with realistic feature distributions
- Feature scaling via StandardScaler (z-score normalization)
- Train/test splitting to evaluate generalization
- Multiple evaluation metrics (RMSE, MAE, R²)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)

# Feature columns used across the pipeline. Keeping this as a module-level
# constant ensures consistency between data generation, preprocessing, and
# any downstream consumers (API, dashboard, etc.).
FEATURE_COLUMNS = ["sqft", "bedrooms", "bathrooms", "age"]

# Price floor — prevents unrealistic negative or near-zero prices that would
# occur when noise overwhelms the signal for small/old properties.
MIN_PRICE = 50_000


def generate_synthetic_data(
    n_samples: int = 1000,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate synthetic housing data with realistic feature distributions.

    The pricing formula models a simplified version of real estate valuation:
        price = base + (sqft × rate) + (bedroom premium) + (bathroom premium)
                - (age depreciation) + gaussian noise

    Args:
        n_samples: Number of data points to generate. Higher values give more
            stable model performance estimates but increase training time.
        seed: Random seed for reproducibility. Using NumPy's modern Generator
            API (default_rng) instead of the legacy RandomState for better
            statistical properties and thread safety.

    Returns:
        DataFrame with columns: sqft, bedrooms, bathrooms, age, price.
    """
    rng = np.random.default_rng(seed)

    # Feature distributions chosen to approximate US housing market ranges
    sqft = rng.integers(800, 5000, n_samples).astype(float)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.integers(1, 4, n_samples)
    age = rng.integers(0, 50, n_samples).astype(float)

    # Gaussian noise simulates unobserved factors (location, renovations,
    # market conditions) that affect price but aren't captured in our features.
    # σ = $20,000 represents roughly 5-10% variation on a median-priced home.
    noise = rng.normal(0, 20_000, n_samples)

    # Linear pricing model with interpretable coefficients:
    #   - $150/sqft: typical suburban rate
    #   - $15K/bedroom: premium for additional bedrooms
    #   - $10K/bathroom: premium for additional bathrooms
    #   - -$1K/year: annual depreciation from age
    price = (
        MIN_PRICE
        + sqft * 150
        + bedrooms * 15_000
        + bathrooms * 10_000
        - age * 1_000
        + noise
    )

    # Clamp to minimum — real houses don't sell for negative prices
    price = np.maximum(price, MIN_PRICE)

    return pd.DataFrame({
        "sqft": sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "price": price.round(0),
    })


def preprocess(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Separate features from target and apply z-score normalization.

    StandardScaler transforms each feature to have mean=0 and std=1. This is
    important because:
    - Linear regression coefficients become comparable across features
    - Gradient-based optimizers converge faster on normalized inputs
    - Tree-based models are scale-invariant, but scaling doesn't hurt them

    Args:
        df: DataFrame containing feature columns and 'price' target.
        scaler: Pre-fitted scaler for inference. If None, a new scaler is
            fit on the provided data (use this for training).

    Returns:
        Tuple of (scaled features array, target array, fitted scaler).
    """
    feature_cols = [c for c in df.columns if c != "price"]

    if scaler is None:
        # Fit a new scaler on training data — learns mean and std per feature
        scaler = StandardScaler()
        X = scaler.fit_transform(df[feature_cols])
    else:
        # Use pre-fitted scaler for test/inference data to prevent data leakage.
        # Data leakage occurs when information from the test set influences
        # the training process, leading to overly optimistic performance estimates.
        X = scaler.transform(df[feature_cols])

    y = df["price"].values
    return X, y, scaler


def get_feature_importance(
    model: Any,
    feature_names: List[str],
) -> Optional[Dict[str, float]]:
    """Extract feature importance from a trained model.

    Different model types expose importance differently:
    - Linear models: absolute values of coefficients (after scaling)
    - Tree ensembles: Gini importance (mean decrease in impurity)

    Args:
        model: A fitted scikit-learn estimator.
        feature_names: Names corresponding to each feature column.

    Returns:
        Dictionary mapping feature names to importance scores, or None
        if the model type doesn't support importance extraction.
    """
    if hasattr(model, "feature_importances_"):
        # Tree-based models (Random Forest, Gradient Boosting)
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # Linear models — use absolute coefficient values as proxy for importance
        importances = np.abs(model.coef_)
    else:
        return None

    # Normalize to sum to 1.0 for easy comparison across models
    total = importances.sum()
    if total > 0:
        importances = importances / total

    return {
        name: round(float(imp), 4)
        for name, imp in zip(feature_names, importances)
    }


def train_and_evaluate(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """Train multiple regression models and compare their performance.

    This function implements the standard ML evaluation workflow:
    1. Split data into train/test sets (stratified by default ratio)
    2. Train each model on the training set only
    3. Evaluate on the held-out test set to estimate generalization

    Metrics explained:
    - **RMSE** (Root Mean Squared Error): Average prediction error in dollars.
      Penalizes large errors more heavily due to squaring. Lower is better.
    - **MAE** (Mean Absolute Error): Average absolute prediction error in dollars.
      More robust to outliers than RMSE. Lower is better.
    - **R²** (Coefficient of Determination): Proportion of variance explained
      by the model. 1.0 = perfect, 0.0 = predicts the mean, <0 = worse than mean.

    Args:
        df: DataFrame with features and 'price' target column.
        test_size: Fraction of data reserved for testing (0.2 = 80/20 split).

    Returns:
        Dictionary mapping model names to their evaluation metrics, including
        RMSE, MAE, R², and feature importance scores.
    """
    X, y, scaler = preprocess(df)

    # Stratification isn't used here because this is regression (continuous target).
    # random_state ensures reproducible splits for fair model comparison.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED
    )

    # Model zoo — each offers different bias-variance tradeoffs:
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100,       # 100 trees in the ensemble
            random_state=RANDOM_SEED,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100,       # 100 boosting stages
            random_state=RANDOM_SEED,
            max_depth=5,            # Limit tree depth to reduce overfitting
            learning_rate=0.1,      # Shrinkage — smaller = more robust but slower
        ),
    }

    results: Dict[str, Any] = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
        mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
        r2 = round(float(r2_score(y_test, y_pred)), 4)

        # Extract which features matter most to this model
        importance = get_feature_importance(model, FEATURE_COLUMNS)

        results[name] = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "feature_importance": importance,
        }

        logger.info(
            "%s — RMSE=$%.0f | MAE=$%.0f | R2=%.4f",
            name, rmse, mae, r2,
        )

    return results
