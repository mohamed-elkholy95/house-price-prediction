"""Overview page — project introduction and model explanations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.title("🏠 House Price Prediction")
st.markdown(
    "Predict house prices using three regression models trained on synthetic "
    "real estate data. This project demonstrates a complete ML pipeline from "
    "data generation through model comparison and deployment."
)

st.markdown("---")

# --- Model descriptions ---
st.subheader("🤖 Models")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### Linear Regression")
    st.markdown(
        "The simplest approach — fits a straight line (hyperplane) through "
        "the feature space. Assumes price changes linearly with each feature.\n\n"
        "**Strengths:** Fast, interpretable coefficients, good baseline\n\n"
        "**Weaknesses:** Can't capture non-linear patterns or feature interactions"
    )

with col2:
    st.markdown("#### Random Forest")
    st.markdown(
        "An ensemble of 100 decision trees, each trained on a random subset "
        "of the data (bagging). Final prediction is the average of all trees.\n\n"
        "**Strengths:** Handles non-linearity, robust to outliers, built-in "
        "feature importance\n\n"
        "**Weaknesses:** Slower to train, less interpretable than linear models"
    )

with col3:
    st.markdown("#### Gradient Boosting")
    st.markdown(
        "Sequential ensemble where each tree corrects errors from the previous "
        "one. Uses gradient descent to minimize the loss function.\n\n"
        "**Strengths:** Often the most accurate, flexible loss functions\n\n"
        "**Weaknesses:** Prone to overfitting, sensitive to hyperparameters, "
        "slowest to train"
    )

st.markdown("---")

# --- Features ---
st.subheader("📋 Features")
st.markdown(
    "The model uses four features to predict house prices. These are generated "
    "synthetically with distributions approximating the US housing market:"
)

feat_col1, feat_col2 = st.columns(2)

with feat_col1:
    st.markdown(
        "- **Square Footage** (800–5,000 sqft): Primary price driver at ~$150/sqft\n"
        "- **Bedrooms** (1–5): Each adds ~$15,000 premium"
    )

with feat_col2:
    st.markdown(
        "- **Bathrooms** (1–3): Each adds ~$10,000 premium\n"
        "- **Property Age** (0–50 years): Depreciates ~$1,000/year"
    )

st.markdown("---")

# --- Metrics explained ---
st.subheader("📏 Evaluation Metrics")
st.markdown(
    "Each model is evaluated using three complementary metrics:"
)

mcol1, mcol2, mcol3 = st.columns(3)

with mcol1:
    st.markdown("#### RMSE")
    st.markdown(
        "**Root Mean Squared Error**\n\n"
        "Average prediction error in dollars, with extra penalty for large "
        "errors (due to squaring). Lower is better."
    )
    st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}")

with mcol2:
    st.markdown("#### MAE")
    st.markdown(
        "**Mean Absolute Error**\n\n"
        "Average absolute prediction error in dollars. More robust to outliers "
        "than RMSE. Lower is better."
    )
    st.latex(r"\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|")

with mcol3:
    st.markdown("#### R²")
    st.markdown(
        "**Coefficient of Determination**\n\n"
        "Proportion of price variance explained by the model. "
        "1.0 = perfect, 0.0 = predicts the mean. Higher is better."
    )
    st.latex(r"R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}")

st.markdown("---")

# --- Quick start ---
st.subheader("🚀 Getting Started")
st.markdown("Navigate to **Train & Compare** to train models, or **Single Prediction** to estimate a specific house price.")
