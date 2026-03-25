"""
House Price Prediction API
===========================

FastAPI application serving the house price prediction pipeline.

Endpoints:
- GET  /health         — Health check
- POST /predict        — Train models and return evaluation metrics
- POST /predict/single — Predict price for a single house
- POST /evaluate/cv    — Cross-validated model evaluation
"""

import logging
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import API_HOST, API_PORT

logger = logging.getLogger(__name__)

app = FastAPI(
    title="House Price Prediction API",
    version="2.0.0",
    description=(
        "ML-powered house price prediction using Linear Regression, "
        "Random Forest, and Gradient Boosting. Supports single-house "
        "predictions, batch evaluation, and cross-validation."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Request body for batch model training and evaluation."""
    n_samples: int = Field(
        default=1000,
        ge=100,
        le=10_000,
        description="Number of synthetic samples to generate for training.",
    )


class SinglePredictRequest(BaseModel):
    """Request body for predicting the price of a single house.

    Each field represents a real estate feature that influences price.
    """
    sqft: float = Field(..., gt=0, le=50_000, description="Square footage of the property")
    bedrooms: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    bathrooms: int = Field(..., ge=0, le=10, description="Number of bathrooms")
    age: float = Field(..., ge=0, le=200, description="Age of the property in years")
    model_name: str = Field(
        default="gradient_boosting",
        description="Model to use: linear_regression, random_forest, or gradient_boosting",
    )


class CVRequest(BaseModel):
    """Request body for cross-validated evaluation."""
    n_samples: int = Field(default=1000, ge=100, le=10_000)
    n_folds: int = Field(default=5, ge=2, le=20)


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring and load balancers."""
    return HealthResponse(status="healthy", version="2.0.0")


@app.post("/predict")
async def predict(req: PredictRequest):
    """Train all models on synthetic data and return evaluation metrics.

    Generates synthetic housing data, trains three regression models,
    and returns RMSE, MAE, R², and feature importance for each.
    """
    from src.price_model import generate_synthetic_data, train_and_evaluate

    df = generate_synthetic_data(n_samples=req.n_samples)
    results = train_and_evaluate(df)
    return {
        "n_samples": len(df),
        "results": results,
    }


@app.post("/predict/single")
async def predict_single(req: SinglePredictRequest):
    """Predict the price of a single house using a specified model.

    This endpoint trains the model on synthetic data (since we don't persist
    trained models in this demo) and returns the predicted price along with
    the model's overall performance metrics.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    from src.config import RANDOM_SEED
    from src.price_model import generate_synthetic_data, preprocess

    valid_models = {
        "linear_regression": LinearRegression,
        "random_forest": lambda: RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_SEED
        ),
        "gradient_boosting": lambda: GradientBoostingRegressor(
            n_estimators=100, random_state=RANDOM_SEED, max_depth=5
        ),
    }

    if req.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown model '{req.model_name}'. "
                   f"Choose from: {list(valid_models.keys())}",
        )

    # Generate training data and fit model
    df = generate_synthetic_data(n_samples=1000)
    X_train, y_train, scaler = preprocess(df)

    model_factory = valid_models[req.model_name]
    model = model_factory() if callable(model_factory) else model_factory
    model.fit(X_train, y_train)

    # Prepare the single input — must use the same scaler as training data
    input_df = pd.DataFrame([{
        "sqft": req.sqft,
        "bedrooms": req.bedrooms,
        "bathrooms": req.bathrooms,
        "age": req.age,
    }])
    X_input = scaler.transform(input_df)

    predicted_price = float(model.predict(X_input)[0])

    return {
        "predicted_price": round(predicted_price, 2),
        "model_used": req.model_name,
        "input": {
            "sqft": req.sqft,
            "bedrooms": req.bedrooms,
            "bathrooms": req.bathrooms,
            "age": req.age,
        },
    }


@app.post("/evaluate/cv")
async def evaluate_cv(req: CVRequest):
    """Run cross-validated evaluation on all models.

    Returns mean and standard deviation of metrics across K folds,
    providing more robust performance estimates than a single split.
    """
    from src.evaluation import cross_validate_models
    from src.price_model import generate_synthetic_data

    df = generate_synthetic_data(n_samples=req.n_samples)
    results = cross_validate_models(df, n_folds=req.n_folds)
    return {
        "n_samples": req.n_samples,
        "n_folds": req.n_folds,
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
