"""Shared test fixtures for the house price prediction test suite.

Fixtures centralize common setup (data generation, model training) so
individual test files stay focused on assertions. Using session-scoped
fixtures for expensive operations avoids redundant computation.
"""

import pytest

from src.price_model import generate_synthetic_data, preprocess, train_and_evaluate


@pytest.fixture(scope="session")
def large_dataset():
    """Generate a large dataset once per test session.

    Session scope means this fixture is created once and shared across
    all tests that request it, avoiding repeated data generation.
    """
    return generate_synthetic_data(1000, seed=42)


@pytest.fixture
def small_dataset():
    """Generate a small dataset for fast unit tests."""
    return generate_synthetic_data(200, seed=99)


@pytest.fixture(scope="session")
def preprocessed_data(large_dataset):
    """Pre-processed feature matrix and target vector."""
    X, y, scaler = preprocess(large_dataset)
    return X, y, scaler


@pytest.fixture(scope="session")
def trained_results(large_dataset):
    """Results from training all models on the large dataset.

    Cached at session scope so multiple test files can inspect model
    results without re-training.
    """
    return train_and_evaluate(large_dataset)
