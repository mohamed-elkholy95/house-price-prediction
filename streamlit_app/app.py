"""Streamlit multi-page app for House Price Prediction.

Navigation:
1. Overview — Project introduction and model descriptions
2. Train & Compare — Batch training with metric visualizations
3. Single Prediction — Predict price for a specific property
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="House Price Prediction",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="expanded",
)

# Custom styling for a clean dark theme
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { background-color: #262730; }
    .stApp { background-color: #0e1117; color: #fff; }
    h1, h2, h3 { color: #1f77b4; }
    </style>
    """,
    unsafe_allow_html=True,
)

pg = st.navigation([
    st.Page("pages/1_📊_Overview.py", title="Overview", icon="📊"),
    st.Page("pages/2_📈_Predict.py", title="Train & Compare", icon="📈"),
    st.Page("pages/3_🏠_Single_Prediction.py", title="Single Prediction", icon="🏠"),
])
pg.run()
