"""Tests for the house price prediction model.

Covers data generation, preprocessing, training, and evaluation with
parametrized tests for edge cases and reproducibility verification.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.price_model import (
    FEATURE_COLUMNS,
    MIN_PRICE,
    generate_synthetic_data,
    get_feature_importance,
    preprocess,
    train_and_evaluate,
)


class TestGenerateSyntheticData:
    """Tests for synthetic data generation."""

    def test_default_shape(self):
        df = generate_synthetic_data()
        assert df.shape == (1000, 5)

    @pytest.mark.parametrize("n_samples", [100, 500, 2000])
    def test_custom_sample_sizes(self, n_samples: int):
        df = generate_synthetic_data(n_samples)
        assert len(df) == n_samples

    def test_column_names(self):
        df = generate_synthetic_data(100)
        expected = ["sqft", "bedrooms", "bathrooms", "age", "price"]
        assert list(df.columns) == expected

    def test_no_missing_values(self):
        df = generate_synthetic_data(500)
        assert df.isna().sum().sum() == 0

    def test_price_floor(self):
        """All prices should be at or above the minimum threshold."""
        df = generate_synthetic_data(2000)
        assert (df["price"] >= MIN_PRICE).all()

    def test_feature_ranges(self):
        """Features should fall within the generation bounds."""
        df = generate_synthetic_data(5000, seed=99)
        assert df["sqft"].between(800, 4999).all()
        assert df["bedrooms"].between(1, 5).all()
        assert df["bathrooms"].between(1, 3).all()
        assert df["age"].between(0, 49).all()

    def test_reproducibility_same_seed(self):
        df1 = generate_synthetic_data(200, seed=42)
        df2 = generate_synthetic_data(200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_data(200, seed=42)
        df2 = generate_synthetic_data(200, seed=99)
        assert not df1.equals(df2)

    def test_price_is_rounded(self):
        """Prices should be whole dollars (rounded during generation)."""
        df = generate_synthetic_data(500)
        assert (df["price"] == df["price"].round(0)).all()


class TestPreprocess:
    """Tests for feature preprocessing and scaling."""

    def test_output_shapes(self):
        df = generate_synthetic_data(200)
        X, y, scaler = preprocess(df)
        assert X.shape == (200, 4)
        assert y.shape == (200,)

    def test_scaled_features_are_standardized(self):
        """After scaling, features should have approximately mean=0, std=1."""
        df = generate_synthetic_data(1000)
        X, _, _ = preprocess(df)
        # Tolerance for finite sample approximation
        assert np.abs(X.mean(axis=0)).max() < 0.1
        assert np.abs(X.std(axis=0) - 1.0).max() < 0.1

    def test_prefitted_scaler(self):
        """Using a pre-fitted scaler should not refit on new data."""
        df_train = generate_synthetic_data(500, seed=1)
        df_test = generate_synthetic_data(100, seed=2)

        _, _, scaler = preprocess(df_train)
        X_test, y_test, same_scaler = preprocess(df_test, scaler=scaler)

        assert X_test.shape == (100, 4)
        assert same_scaler is scaler  # Same object, not refitted

    def test_returns_standard_scaler(self):
        df = generate_synthetic_data(100)
        _, _, scaler = preprocess(df)
        assert isinstance(scaler, StandardScaler)


class TestTrainAndEvaluate:
    """Tests for model training and evaluation pipeline."""

    def test_returns_all_models(self):
        df = generate_synthetic_data(200)
        results = train_and_evaluate(df)
        expected_models = {"linear_regression", "random_forest", "gradient_boosting"}
        assert set(results.keys()) == expected_models

    def test_metrics_present(self):
        df = generate_synthetic_data(200)
        results = train_and_evaluate(df)
        for name, metrics in results.items():
            assert "rmse" in metrics, f"{name} missing rmse"
            assert "mae" in metrics, f"{name} missing mae"
            assert "r2" in metrics, f"{name} missing r2"
            assert "feature_importance" in metrics, f"{name} missing feature_importance"

    def test_r2_is_positive(self):
        """All models should beat random guessing on this synthetic data."""
        df = generate_synthetic_data(500)
        results = train_and_evaluate(df)
        for name, metrics in results.items():
            assert metrics["r2"] > 0, f"{name} has non-positive R²"

    def test_rmse_is_reasonable(self):
        """RMSE should be well below the price range for good models."""
        df = generate_synthetic_data(1000)
        price_range = df["price"].max() - df["price"].min()
        results = train_and_evaluate(df)
        for name, metrics in results.items():
            assert metrics["rmse"] < price_range * 0.5, (
                f"{name} RMSE ({metrics['rmse']}) too high relative to price range"
            )

    def test_feature_importance_sums_to_one(self):
        df = generate_synthetic_data(500)
        results = train_and_evaluate(df)
        for name, metrics in results.items():
            importance = metrics["feature_importance"]
            if importance is not None:
                total = sum(importance.values())
                assert abs(total - 1.0) < 0.01, (
                    f"{name} importance sums to {total}, expected ~1.0"
                )

    @pytest.mark.parametrize("test_size", [0.1, 0.2, 0.3])
    def test_different_split_ratios(self, test_size: float):
        """Model should train successfully with various split ratios."""
        df = generate_synthetic_data(300)
        results = train_and_evaluate(df, test_size=test_size)
        assert len(results) == 3


class TestGetFeatureImportance:
    """Tests for the feature importance extraction utility."""

    def test_with_linear_model(self):
        from sklearn.linear_model import LinearRegression
        df = generate_synthetic_data(200)
        X, y, _ = preprocess(df)
        model = LinearRegression().fit(X, y)
        importance = get_feature_importance(model, FEATURE_COLUMNS)
        assert importance is not None
        assert set(importance.keys()) == set(FEATURE_COLUMNS)

    def test_with_unsupported_model(self):
        """Models without coef_ or feature_importances_ should return None."""

        class DummyModel:
            pass

        result = get_feature_importance(DummyModel(), FEATURE_COLUMNS)
        assert result is None
