import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import streamlit as st
from src.price_model import generate_synthetic_data, train_and_evaluate
n = st.slider("Samples", 100, 5000, 1000)
if st.button("Train & Evaluate", type="primary"):
    with st.spinner("Training..."):
        df = generate_synthetic_data(n)
        results = train_and_evaluate(df)
    for name, m in results.items():
        st.subheader(name.replace("_", " ").title())
        st.json(m)
