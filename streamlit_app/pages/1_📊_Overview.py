import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
st.title("🏠 House Price Prediction")
st.markdown("Predict house prices using ML models with synthetic real estate data.")
col1, col2 = st.columns(2)
with col1: st.subheader("Models"); st.markdown("- Linear Regression\n- Random Forest\n- Gradient Boosting")
with col2: st.subheader("Features"); st.markdown("- Sqft, Bedrooms, Bathrooms\n- Age of property")
