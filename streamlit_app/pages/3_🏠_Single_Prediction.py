"""Single House Prediction — estimate the price of a specific property.

This page allows users to input individual house features and get an
instant price estimate from any of the three trained models. It also
shows how the input compares to the training data distribution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.config import RANDOM_SEED
from src.price_model import generate_synthetic_data, preprocess

st.title("🏠 Single House Price Prediction")
st.markdown(
    "Enter house features below to get a price estimate. The model trains on "
    "synthetic data each time to demonstrate the full prediction pipeline."
)

# --- Input form ---
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Square Footage", min_value=400, max_value=10000, value=2000, step=100)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)

with col2:
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=6, value=2)
    age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=10)

model_choice = st.selectbox(
    "Model",
    ["gradient_boosting", "random_forest", "linear_regression"],
    format_func=lambda x: x.replace("_", " ").title(),
)

if st.button("💰 Predict Price", type="primary"):
    with st.spinner("Training model and predicting..."):
        # Generate training data
        df = generate_synthetic_data(1000)
        X_train, y_train, scaler = preprocess(df)

        # Initialize and train model
        model_map = {
            "linear_regression": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=RANDOM_SEED, max_depth=5
            ),
        }
        model = model_map[model_choice]
        model.fit(X_train, y_train)

        # Predict using the same scaler
        input_df = pd.DataFrame([{
            "sqft": sqft, "bedrooms": bedrooms,
            "bathrooms": bathrooms, "age": age,
        }])
        X_input = scaler.transform(input_df)
        predicted_price = model.predict(X_input)[0]

    # --- Display results ---
    st.markdown("---")
    st.subheader("Predicted Price")

    pcol1, pcol2, pcol3 = st.columns([2, 1, 1])
    with pcol1:
        st.metric(
            label=f"Estimated Value ({model_choice.replace('_', ' ').title()})",
            value=f"${predicted_price:,.0f}",
        )

    # Show where this house falls in the training data distribution
    st.subheader("📊 How Does This Compare?")
    st.markdown("Position of your predicted price within the training data distribution:")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["price"], nbinsx=50,
        name="Training Data Prices",
        marker_color="#636EFA",
        opacity=0.7,
    ))
    fig.add_vline(
        x=predicted_price,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Your house: ${predicted_price:,.0f}",
        annotation_position="top",
    )
    fig.update_layout(
        title="Your Prediction vs. Training Data",
        xaxis_title="House Price ($)",
        yaxis_title="Count",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Percentile ranking
    percentile = (df["price"] < predicted_price).mean() * 100
    st.info(
        f"This property would rank at the **{percentile:.0f}th percentile** "
        f"of the training data — {'above' if percentile > 50 else 'below'} "
        f"the median price of ${df['price'].median():,.0f}."
    )
