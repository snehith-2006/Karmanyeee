# Predicting Campaign Budget in Digital Marketing

This repository contains a machine learning project designed to predict optimal campaign expenditure (`spend_usd`) based on digital marketing performance metrics. The project features a complete pipeline from data exploration and feature engineering to a real-time web interface powered by **Flask**.

---

## 🚀 Features
* **Machine Learning Pipeline**: Automated data cleaning and feature engineering (CTR, CPC, Conversion Rate).
* **High-Performance Model**: Utilizes a **Random Forest Regressor** with 300 estimators, achieving an $R^2$ Score of **0.999**.
* [cite_start]**Web Dashboard**: An interactive Flask application that allows users to input metrics and receive instant budget suggestions.
* [cite_start]**Dual Execution Modes**: Supports both a live ML model mode (using `joblib`) and a deterministic mock mode for development.

---

## 🛠️ Technical Stack
* **Language**: Python 3.11
* **Libraries**: `Flask`, `scikit-learn`, `pandas`, `numpy`, `joblib`
* **Model**: Random Forest Regressor

---

## 📂 Project Structure
```text
├── app2.py                # Main Flask application logic
├── budget_model.pkl       # Serialized Random Forest model
├── Dessertation (1).ipynb # Jupyter Notebook for training and EDA
└── requirements.txt       # List of Python dependencies
