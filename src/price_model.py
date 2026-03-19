"""WORK IN PROGRESS — Core structure and imports."""

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
