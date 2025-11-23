# ₿ Short-Term Bitcoin Price Prediction — Deep Learning & Time-Series Modeling
This project focuses on forecasting short-term Bitcoin closing prices using LSTM Neural Networks trained on high-frequency minute-level BTC-USD market data. It covers the full end-to-end data science workflow — from data preprocessing and exploratory analysis to sequence modeling, evaluation, and insights.

# 🚀 Project Overview
The goal of this project is to build a robust and reliable deep learning model capable of predicting short-term Bitcoin price movements by learning sequential patterns from historical data.
The model captures Bitcoin's volatility, temporal dependencies, and rapid market fluctuations using LSTM layers optimized for time-series forecasting.

## This project follows a structured pipeline:
Data Loading & Cleaning: Handling missing values, converting timestamps, filtering anomalies.

EDA: Understanding price trends, volatility patterns, correlations, and distribution of returns.

Feature Engineering: Scaling, creating time sequences, selecting key predictors (Open, High, Low, Volume).

Modeling: Designing and training an LSTM model with sequence learning capabilities.

Evaluation: Using RMSE, MAE, MAPE, and visual comparison of actual vs. predicted prices.

Insights: Interpreting model performance and understanding market behavior.

# 📂 Key Features
Cleaned and chronologically ordered minute-level dataset
Advanced EDA: volatility analysis, rolling statistics, return distributions
Correlation heatmaps and price movement visualizations
LSTM-based deep learning model for sequence forecasting
Hyperparameter tuning and model optimization
Comprehensive performance evaluation (RMSE, MAE, R²)
Clear insights into Bitcoin’s temporal price patterns

# 🧰 Tech Stack

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-Learn

TensorFlow / Keras (LSTM)

SQLite (optional for data management)

VS Code / Jupyter Notebook

# 📊 Results
Achieved high prediction accuracy with strong alignment between actual and predicted values
LSTM model successfully captured key trends and short-term fluctuations
Demonstrated effectiveness of sequence modeling on volatile financial time-series data
Provided insights into Bitcoin’s price dynamics and market volatility patterns

Processed 525K+ Kaggle records and achieved R² = 0.941, RMSE = 432.31, MAE = 347.14, 
effectively modeling high-frequency volatility.

# 📁 Dataset
The project uses a minute-level Bitcoin market dataset (BTC-USD) sourced from Kaggle, containing:

Timestamp (UNIX & datetime)

Open, High, Low, Close

Volume (BTC & USD)

Over 525,000+ rows for the year 2017

This high-frequency dataset is ideal for short-term forecasting and sequence modeling.

# 🤝 Contributions
Contributions, improvements, and suggestions are always welcome!
Feel free to open an issue or submit a pull request.
