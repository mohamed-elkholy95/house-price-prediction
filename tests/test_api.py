"""Tests for the FastAPI house price prediction API.

Covers all endpoints including validation, error handling, and edge cases.
"""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for the GET /health endpoint."""

    def test_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_healthy_status(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_returns_version(self):
        data = client.get("/health").json()
        assert "version" in data


class TestPredictEndpoint:
    """Tests for the POST /predict batch evaluation endpoint."""

    def test_default_samples(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 1000
        assert "results" in data

    def test_custom_samples(self):
        resp = client.post("/predict", json={"n_samples": 200})
        assert resp.status_code == 200
        assert resp.json()["n_samples"] == 200

    def test_results_contain_all_models(self):
        resp = client.post("/predict", json={"n_samples": 200})
        results = resp.json()["results"]
        assert "linear_regression" in results
        assert "random_forest" in results
        assert "gradient_boosting" in results

    def test_too_few_samples_rejected(self):
        """n_samples below 100 should be rejected by validation."""
        resp = client.post("/predict", json={"n_samples": 10})
        assert resp.status_code == 422

    def test_too_many_samples_rejected(self):
        """n_samples above 10000 should be rejected by validation."""
        resp = client.post("/predict", json={"n_samples": 100_000})
        assert resp.status_code == 422


class TestSinglePredictEndpoint:
    """Tests for the POST /predict/single endpoint."""

    def test_basic_prediction(self):
        resp = client.post("/predict/single", json={
            "sqft": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 10,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_price" in data
        assert data["predicted_price"] > 0

    def test_returns_model_used(self):
        resp = client.post("/predict/single", json={
            "sqft": 1500,
            "bedrooms": 2,
            "bathrooms": 1,
            "age": 5,
            "model_name": "random_forest",
        })
        data = resp.json()
        assert data["model_used"] == "random_forest"

    def test_returns_input_echo(self):
        """Response should echo back the input features for verification."""
        resp = client.post("/predict/single", json={
            "sqft": 3000,
            "bedrooms": 4,
            "bathrooms": 3,
            "age": 20,
        })
        data = resp.json()
        assert data["input"]["sqft"] == 3000
        assert data["input"]["bedrooms"] == 4

    def test_invalid_model_rejected(self):
        resp = client.post("/predict/single", json={
            "sqft": 2000,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 10,
            "model_name": "neural_network",
        })
        assert resp.status_code == 400

    def test_negative_sqft_rejected(self):
        resp = client.post("/predict/single", json={
            "sqft": -100,
            "bedrooms": 3,
            "bathrooms": 2,
            "age": 10,
        })
        assert resp.status_code == 422

    def test_missing_required_fields(self):
        resp = client.post("/predict/single", json={"sqft": 2000})
        assert resp.status_code == 422


class TestCVEndpoint:
    """Tests for the POST /evaluate/cv cross-validation endpoint."""

    def test_basic_cv(self):
        resp = client.post("/evaluate/cv", json={
            "n_samples": 200,
            "n_folds": 3,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_folds"] == 3
        assert "results" in data

    def test_default_values(self):
        resp = client.post("/evaluate/cv", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 1000
        assert data["n_folds"] == 5
