"""Quick start example for 12-house-price-prediction.

Run: python examples/quickstart.py
"""
import sys
sys.path.insert(0, ".")

from src.config import RANDOM_SEED

print(f"=== 12 House Price Prediction Quick Start ===")
print(f"Random seed: {RANDOM_SEED}")
print("Run the Streamlit app for the full interactive experience:")
print("  streamlit run streamlit_app/app.py")
