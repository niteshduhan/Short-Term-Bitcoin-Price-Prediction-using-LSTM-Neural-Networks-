# ₿ Bitcoin Price Forecasting — LSTM

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Preprocessing-F7931E?style=flat-square&logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> Short-term BTC-USD closing price forecasting using sequence-based deep learning on 525K+ minute-level records.

---

## Overview

Trained an LSTM model on high-frequency Bitcoin market data (minute-level, 2017) to predict short-term closing prices.

| Metric | Value |
|--------|-------|
| R² Score | **0.941** |
| RMSE | **432.31** |
| MAE | **347.14** |
| Dataset Size | **525K+ rows** |
| Granularity | **1-minute intervals** |

Model captures sequential volatility patterns across Open, High, Low, Close, and Volume features — outperforming naive and ARIMA baselines on short-horizon forecasts.

---

## Methodology
```
1. Data Loading       → Parse UNIX timestamps, handle nulls, enforce chronological order
2. EDA                → Rolling stats, return distributions, volatility analysis, correlation heatmaps
3. Feature Engineering → Min-Max scaling, sequence windowing (look-back = N steps), train/test split
4. Model              → Stacked LSTM layers with Dropout, trained on sequence-to-point mapping
5. Evaluation         → RMSE, MAE, R², visual overlay of actual vs. predicted
```

---

## Results

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| LSTM (this work) | **432.31** | **347.14** | **0.941** |
| ARIMA (baseline) | ~697.40 | ~541.20 | ~0.783 |

LSTM closed ~38% tighter on RMSE vs. ARIMA baseline on the held-out test window.

---

## Project Structure
```
bitcoin-lstm-forecasting/
├── data/
│   └── btcusd_1min_2017.csv        # Raw Kaggle dataset (525K+ rows)
├── notebooks/
│   ├── 01_eda.ipynb                 # EDA, volatility, correlation analysis
│   ├── 02_preprocessing.ipynb       # Scaling, sequence generation
│   └── 03_lstm_model.ipynb          # Model training & evaluation
├── src/
│   ├── preprocess.py                # Feature engineering pipeline
│   ├── model.py                     # LSTM architecture
│   └── evaluate.py                  # Metrics + plotting
├── outputs/
│   ├── predictions.csv
│   └── loss_curves.png
├── requirements.txt
└── README.md
```

---

## How to Run
```bash
# Clone
git clone https://github.com/niteshduhan/bitcoin-lstm-forecasting.git
cd bitcoin-lstm-forecasting

# Install dependencies
pip install -r requirements.txt

# Run EDA
jupyter notebook notebooks/01_eda.ipynb

# Train model
jupyter notebook notebooks/03_lstm_model.ipynb
```

---

## Key Takeaways

- **Sequence modeling on financial data**: Demonstrates LSTM's ability to learn temporal dependencies in noisy, high-frequency market data — R² of 0.941 on unseen test data.
- **End-to-end ML pipeline**: Covers the full workflow from raw tick data to evaluation-ready predictions with reproducible preprocessing and architecture choices.
- **Volatility-aware forecasting**: Rolling stats and return distribution analysis during EDA directly informed feature selection and scaling strategy.

---

## Dataset

- **Source**: [Kaggle — Bitcoin Historical Data](https://www.kaggle.com/)
- **Coverage**: Full year 2017, 1-minute granularity
- **Columns**: `Timestamp`, `Open`, `High`, `Low`, `Close`, `Volume (BTC)`, `Volume (USD)`
- **Size**: 525,000+ rows

---

## Author

**Nitesh Duhan** — Data Scientist  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-niteshduhan--carp112-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/niteshduhan-carp112)
[![Gmail](https://img.shields.io/badge/Gmail-niteshduhan686@gmail.com-red?style=flat-square&logo=gmail)](mailto:niteshduhan686@gmail.com)

---

> ⭐ If this project helped you — star the repo. It's the highest-signal feedback.
