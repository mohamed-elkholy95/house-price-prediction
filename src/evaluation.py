"""
Model Evaluation & Cross-Validation
====================================

This module provides robust evaluation strategies beyond simple train/test splits.

Why cross-validation matters:
- A single train/test split can produce misleading results depending on which
  samples end up in which set (high variance in estimates).
- K-fold cross-validation rotates through K different splits, giving K estimates
  that we average for a more stable performance picture.
- The standard deviation across folds reveals how sensitive the model is to
  the particular data it sees (model stability).

This module also implements learning curves, which help diagnose whether a model
is suffering from high bias (underfitting) or high variance (overfitting) by
plotting performance against training set size.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, learning_curve

from src.config import RANDOM_SEED
from src.price_model import preprocess

logger = logging.getLogger(__name__)


def rmse_scorer():
    """Create a scikit-learn scorer for RMSE.

    scikit-learn's scoring convention expects higher = better, so we negate
    the MSE (neg_mean_squared_error). We then take the sqrt of the absolute
    value to get RMSE.
    """
    return make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    )


def cross_validate_models(
    df: pd.DataFrame,
    n_folds: int = 5,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate models using K-fold cross-validation.

    K-fold works by:
    1. Splitting data into K equal-sized folds
    2. For each fold i: train on all folds except i, test on fold i
    3. Average the K test scores for a robust performance estimate

    Args:
        df: DataFrame with features and 'price' target column.
        n_folds: Number of cross-validation folds. 5 is standard; 10 gives
            slightly less biased estimates but takes 2x longer to compute.

    Returns:
        Dictionary mapping model names to CV results including mean and
        standard deviation of each metric across folds.
    """
    X, y, _ = preprocess(df)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_SEED, max_depth=5
        ),
    }

    # Scoring dictionary — sklearn computes all metrics in a single CV run
    scoring = {
        "r2": "r2",
        "neg_rmse": rmse_scorer(),
        "neg_mae": "neg_mean_absolute_error",
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        cv_results = cross_validate(
            model, X, y,
            cv=n_folds,
            scoring=scoring,
            return_train_score=True,  # Helps detect overfitting
        )

        # Extract and format results. sklearn prefixes with 'test_' and 'train_'
        test_r2 = cv_results["test_r2"]
        test_rmse = -cv_results["test_neg_rmse"]  # Negate back to positive
        test_mae = -cv_results["test_neg_mae"]
        train_r2 = cv_results["train_r2"]

        results[name] = {
            "r2_mean": round(float(test_r2.mean()), 4),
            "r2_std": round(float(test_r2.std()), 4),
            "rmse_mean": round(float(test_rmse.mean()), 2),
            "rmse_std": round(float(test_rmse.std()), 2),
            "mae_mean": round(float(test_mae.mean()), 2),
            "mae_std": round(float(test_mae.std()), 2),
            "train_r2_mean": round(float(train_r2.mean()), 4),
            # Gap between train and test R² indicates overfitting
            "overfit_gap": round(float(train_r2.mean() - test_r2.mean()), 4),
            "fit_time_mean": round(float(cv_results["fit_time"].mean()), 3),
        }

        logger.info(
            "%s CV — R²=%.4f±%.4f | RMSE=$%,.0f±$%,.0f | overfit_gap=%.4f",
            name,
            results[name]["r2_mean"],
            results[name]["r2_std"],
            results[name]["rmse_mean"],
            results[name]["rmse_std"],
            results[name]["overfit_gap"],
        )

    return results


def compute_learning_curve(
    df: pd.DataFrame,
    model_name: str = "gradient_boosting",
    n_points: int = 8,
) -> Dict[str, Any]:
    """Compute a learning curve to diagnose bias vs. variance.

    A learning curve plots model performance against training set size:
    - If train and test scores are both low → high bias (underfitting)
      → need a more complex model or better features
    - If train score is high but test score is low → high variance (overfitting)
      → need more data, regularization, or simpler model
    - If both converge to a high score → good fit

    Args:
        df: DataFrame with features and target.
        model_name: Which model to analyze.
        n_points: Number of training set sizes to evaluate.

    Returns:
        Dictionary with train_sizes, train_scores, and test_scores arrays.
    """
    X, y, _ = preprocess(df)

    model_map = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=50, random_state=RANDOM_SEED
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=50, random_state=RANDOM_SEED, max_depth=5
        ),
    }

    model = model_map.get(model_name)
    if model is None:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose from: {list(model_map.keys())}"
        )

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, n_points),
        cv=5,
        scoring="r2",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    return {
        "model": model_name,
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": np.mean(train_scores, axis=1).round(4).tolist(),
        "train_scores_std": np.std(train_scores, axis=1).round(4).tolist(),
        "test_scores_mean": np.mean(test_scores, axis=1).round(4).tolist(),
        "test_scores_std": np.std(test_scores, axis=1).round(4).tolist(),
    }
