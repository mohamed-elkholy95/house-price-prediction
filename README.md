<div align="center">

# 🏠 House Price Prediction

**Regression pipeline** with linear regression, random forest, and gradient boosting

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-38%20passed-success?style=flat-square)](#testing)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

A complete **house price prediction pipeline** demonstrating core ML engineering skills: data generation, feature preprocessing, model training, evaluation, and deployment via REST API and interactive dashboard.

Three regression models are compared side-by-side:
- **Linear Regression** — Interpretable baseline with coefficient analysis
- **Random Forest** — Ensemble bagging with built-in feature importance
- **Gradient Boosting** — Sequential boosting for maximum accuracy

## Features

- 📊 **Synthetic Data Generation** — Realistic housing features (sqft, bedrooms, bathrooms, age) with noise-augmented pricing
- 🏆 **3 Regression Models** — Linear, Random Forest, and Gradient Boosting with full comparison
- 📏 **Evaluation Metrics** — RMSE, MAE, R² with feature importance analysis
- 🔄 **Cross-Validation** — K-fold CV with overfitting detection and learning curves
- 📋 **Interactive Dashboard** — Streamlit app with Plotly visualizations, single-house prediction, and data exploration
- 🚀 **REST API** — FastAPI endpoints for batch evaluation, single prediction, and cross-validation
- ✅ **Comprehensive Tests** — 38+ tests with parametrized edge cases and shared fixtures

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Launch dashboard
streamlit run streamlit_app/app.py

# Start API server
python -m src.api.main
```

## Project Structure

```
├── src/
│   ├── config.py          # Configuration with env var overrides
│   ├── price_model.py     # Data generation, preprocessing, training
│   ├── evaluation.py      # Cross-validation and learning curves
│   └── api/
│       └── main.py        # FastAPI endpoints
├── streamlit_app/
│   ├── app.py             # Multi-page Streamlit entry point
│   └── pages/
│       ├── 1_Overview.py       # Model explanations and metrics guide
│       ├── 2_Predict.py        # Train & compare with visualizations
│       └── 3_Single_Prediction.py  # Individual house price estimation
├── tests/
│   ├── conftest.py        # Shared session-scoped fixtures
│   ├── test_price.py      # Model and data generation tests
│   ├── test_evaluation.py # Cross-validation and learning curve tests
│   └── test_api.py        # API endpoint tests with validation
├── docs/
│   ├── ARCHITECTURE.md    # System design decisions
│   ├── DEVELOPMENT.md     # Development setup guide
│   └── CONTRIBUTING.md    # Contribution guidelines
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with version info |
| `POST` | `/predict` | Train all models, return metrics |
| `POST` | `/predict/single` | Predict price for one house |
| `POST` | `/evaluate/cv` | K-fold cross-validation results |

### Example: Single Prediction

```bash
curl -X POST http://localhost:8012/predict/single \
  -H "Content-Type: application/json" \
  -d '{"sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age": 10}'
```

```json
{
  "predicted_price": 395234.50,
  "model_used": "gradient_boosting",
  "input": {"sqft": 2000, "bedrooms": 3, "bathrooms": 2, "age": 10}
}
```

## Configuration

All settings support environment variable overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `RANDOM_SEED` | `42` | Reproducibility seed |
| `API_HOST` | `0.0.0.0` | API bind address |
| `API_PORT` | `8012` | API listen port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEFAULT_N_SAMPLES` | `1000` | Default training set size |

## Testing

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --tb=short

# Run specific test class
python -m pytest tests/test_price.py::TestGenerateSyntheticData -v
```

Test coverage includes:
- **Data generation**: Shape, ranges, reproducibility, price floor
- **Preprocessing**: Scaling verification, data leakage prevention
- **Model training**: All metrics present, R² positivity, feature importance
- **API endpoints**: Happy paths, validation errors, edge cases
- **Cross-validation**: Fold counts, metric stability, overfitting detection

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
