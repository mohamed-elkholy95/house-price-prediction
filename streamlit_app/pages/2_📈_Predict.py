"""Train & Compare — interactive model training and visualization page.

This page allows users to:
1. Configure sample size and train all three models
2. Compare metrics side-by-side in a bar chart
3. View feature importance rankings per model
4. Explore the generated data distribution
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.price_model import generate_synthetic_data, train_and_evaluate

st.title("📈 Train & Compare Models")
st.markdown(
    "Configure sample size, train all three regression models, and compare "
    "their performance metrics and feature importance rankings."
)

# --- Sidebar controls ---
n = st.slider("Training Samples", 100, 5000, 1000, step=100)

if st.button("🚀 Train & Evaluate", type="primary"):
    with st.spinner("Generating data and training models..."):
        df = generate_synthetic_data(n)
        results = train_and_evaluate(df)

    st.success(f"Trained on {n:,} samples")

    # --- Metric comparison bar chart ---
    st.subheader("Model Comparison")

    model_names = list(results.keys())
    display_names = [name.replace("_", " ").title() for name in model_names]

    # R² comparison
    r2_values = [results[m]["r2"] for m in model_names]
    fig_r2 = px.bar(
        x=display_names, y=r2_values,
        labels={"x": "Model", "y": "R² Score"},
        title="R² Score by Model (higher is better)",
        color=display_names,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_r2.update_layout(showlegend=False, yaxis_range=[0, 1])
    st.plotly_chart(fig_r2, use_container_width=True)

    # RMSE and MAE comparison
    col1, col2 = st.columns(2)

    with col1:
        rmse_values = [results[m]["rmse"] for m in model_names]
        fig_rmse = px.bar(
            x=display_names, y=rmse_values,
            labels={"x": "Model", "y": "RMSE ($)"},
            title="RMSE by Model (lower is better)",
            color=display_names,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_rmse.update_layout(showlegend=False)
        st.plotly_chart(fig_rmse, use_container_width=True)

    with col2:
        mae_values = [results[m]["mae"] for m in model_names]
        fig_mae = px.bar(
            x=display_names, y=mae_values,
            labels={"x": "Model", "y": "MAE ($)"},
            title="MAE by Model (lower is better)",
            color=display_names,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_mae.update_layout(showlegend=False)
        st.plotly_chart(fig_mae, use_container_width=True)

    # --- Feature importance ---
    st.subheader("Feature Importance")
    st.markdown(
        "Shows which features each model relies on most. Linear Regression uses "
        "absolute coefficient values; tree models use Gini importance (mean "
        "decrease in impurity)."
    )

    importance_cols = st.columns(len(model_names))
    for col, name, display in zip(importance_cols, model_names, display_names):
        with col:
            importance = results[name].get("feature_importance")
            if importance:
                fig_imp = px.bar(
                    x=list(importance.values()),
                    y=list(importance.keys()),
                    orientation="h",
                    title=display,
                    labels={"x": "Importance", "y": "Feature"},
                    color=list(importance.keys()),
                    color_discrete_sequence=px.colors.qualitative.Bold,
                )
                fig_imp.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig_imp, use_container_width=True)

    # --- Detailed metrics table ---
    st.subheader("Detailed Metrics")
    for name, metrics in results.items():
        with st.expander(name.replace("_", " ").title()):
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("R²", f"{metrics['r2']:.4f}")
            mcol2.metric("RMSE", f"${metrics['rmse']:,.0f}")
            mcol3.metric("MAE", f"${metrics['mae']:,.0f}")

    # --- Data exploration ---
    st.subheader("📊 Training Data Distribution")

    fig_hist = px.histogram(
        df, x="price", nbins=50,
        title="Price Distribution",
        labels={"price": "House Price ($)", "count": "Count"},
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    # Scatter matrix of features vs price
    fig_scatter = px.scatter_matrix(
        df,
        dimensions=["sqft", "bedrooms", "bathrooms", "age"],
        color="price",
        title="Feature Scatter Matrix (colored by price)",
        color_continuous_scale="Viridis",
        height=600,
    )
    fig_scatter.update_traces(diagonal_visible=False, marker=dict(size=3))
    st.plotly_chart(fig_scatter, use_container_width=True)
