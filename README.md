<div align="center">

# 🏠 House Price Prediction

**Regression pipeline** with linear regression, random forest, and gradient boosting

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat-square&logo=scikit-learn)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-7%20passed-success?style=flat-square)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat-square)](https://streamlit.io)

</div>

## Overview

A **house price prediction pipeline** using three regression models: Linear Regression, Random Forest, and Gradient Boosting. Generates synthetic real estate data (sqft, bedrooms, bathrooms, age) and evaluates models using RMSE, MAE, and R² metrics.

## Features

- 📊 **Synthetic Data** — Realistic housing features with noise-augmented pricing
- 🏆 **3 Regression Models** — Linear, Random Forest, Gradient Boosting
- 📏 **Evaluation Metrics** — RMSE, MAE, R² with model comparison
- 📋 **Interactive Dashboard** — Train and compare models visually
- 🚀 **REST API** — Prediction endpoint with configurable data size

## Quick Start

```bash
git clone https://github.com/mohamed-elkholy95/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
python -m pytest tests/ -v
streamlit run streamlit_app/app.py
```

## Author

**Mohamed Elkholy** — [GitHub](https://github.com/mohamed-elkholy95) · melkholy@techmatrix.com
