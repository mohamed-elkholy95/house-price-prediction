"""WORK IN PROGRESS — Adding methods and implementation details."""

"""House price prediction model."""
import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.config import RANDOM_SEED

logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 1000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sqft = rng.integers(800, 5000, n_samples).astype(float)
    bedrooms = rng.integers(1, 6, n_samples)
    bathrooms = rng.integers(1, 4, n_samples)
    age = rng.integers(0, 50, n_samples).astype(float)
    noise = rng.normal(0, 20000, n_samples)
    price = 50000 + sqft * 150 + bedrooms * 15000 + bathrooms * 10000 - age * 1000 + noise
    price = np.maximum(price, 50000)
    return pd.DataFrame({"sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age": age, "price": price.round(0)})


def preprocess(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    feature_cols = [c for c in df.columns if c != "price"]
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    y = df["price"].values
    return X, y, scaler

