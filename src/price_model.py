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


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    X, y, _ = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED)
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=RANDOM_SEED, max_depth=5),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        mae = round(mean_absolute_error(y_test, y_pred), 2)
        r2 = round(r2_score(y_test, y_pred), 4)
        results[name] = {"rmse": rmse, "mae": mae, "r2": r2}
        logger.info("%s: RMSE=%.0f, R2=%.4f", name, rmse, r2)
    return results
