"""Tests for the cross-validation and learning curve module."""

import pytest

from src.evaluation import compute_learning_curve, cross_validate_models
from src.price_model import generate_synthetic_data


class TestCrossValidation:
    """Tests for K-fold cross-validation."""

    @pytest.fixture
    def sample_data(self):
        return generate_synthetic_data(300)

    def test_returns_all_models(self, sample_data):
        results = cross_validate_models(sample_data, n_folds=3)
        expected = {"linear_regression", "random_forest", "gradient_boosting"}
        assert set(results.keys()) == expected

    def test_cv_metrics_present(self, sample_data):
        results = cross_validate_models(sample_data, n_folds=3)
        required_keys = {
            "r2_mean", "r2_std", "rmse_mean", "rmse_std",
            "mae_mean", "mae_std", "train_r2_mean", "overfit_gap",
            "fit_time_mean",
        }
        for name, metrics in results.items():
            assert required_keys.issubset(metrics.keys()), (
                f"{name} missing keys: {required_keys - set(metrics.keys())}"
            )

    def test_r2_mean_is_positive(self, sample_data):
        results = cross_validate_models(sample_data, n_folds=3)
        for name, metrics in results.items():
            assert metrics["r2_mean"] > 0, f"{name} CV R² should be positive"

    def test_std_is_non_negative(self, sample_data):
        results = cross_validate_models(sample_data, n_folds=3)
        for name, metrics in results.items():
            assert metrics["r2_std"] >= 0
            assert metrics["rmse_std"] >= 0

    @pytest.mark.parametrize("n_folds", [3, 5])
    def test_different_fold_counts(self, sample_data, n_folds):
        results = cross_validate_models(sample_data, n_folds=n_folds)
        assert len(results) == 3


class TestLearningCurve:
    """Tests for learning curve computation."""

    @pytest.fixture
    def sample_data(self):
        return generate_synthetic_data(300)

    def test_returns_expected_keys(self, sample_data):
        result = compute_learning_curve(sample_data, model_name="linear_regression", n_points=4)
        expected_keys = {
            "model", "train_sizes", "train_scores_mean",
            "train_scores_std", "test_scores_mean", "test_scores_std",
        }
        assert set(result.keys()) == expected_keys

    def test_correct_number_of_points(self, sample_data):
        result = compute_learning_curve(sample_data, model_name="linear_regression", n_points=4)
        assert len(result["train_sizes"]) == 4
        assert len(result["train_scores_mean"]) == 4

    def test_train_sizes_are_increasing(self, sample_data):
        result = compute_learning_curve(sample_data, model_name="linear_regression", n_points=5)
        sizes = result["train_sizes"]
        assert sizes == sorted(sizes)

    def test_invalid_model_raises(self, sample_data):
        with pytest.raises(ValueError, match="Unknown model"):
            compute_learning_curve(sample_data, model_name="invalid_model")

    @pytest.mark.parametrize("model_name", [
        "linear_regression", "random_forest", "gradient_boosting",
    ])
    def test_all_models_supported(self, sample_data, model_name):
        result = compute_learning_curve(sample_data, model_name=model_name, n_points=3)
        assert result["model"] == model_name
