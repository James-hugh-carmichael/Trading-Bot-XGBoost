# Trading Bot with XGBoost Classifier & Regressor

A Python-based trading bot that uses machine learning (XGBoost classification and regression models) to predict stock price movements and execute trades automatically via the Alpaca API.

---

## Features

- **Feature engineering**: technical indicators and rolling statistics computed from historical price data
- **Dual ML models**: XGBoost classifier (direction) and regressor (magnitude) trained on historical stock data
- **Trade signal generation**: automated long and short position signals
- **Performance evaluation**: cumulative returns, win rate, and other key metrics
- **Visualization**: model predictions and feature importance plots

---

## Getting Started

### Requirements

- Python 3.8+
- `pandas`
- `numpy`
- `matplotlib`
- `xgboost`
- `ta` (technical analysis library)
- `joblib`
- `yfinance`
- `alpaca-trade-api`

### Installation

```bash
pip install -r requirements.txt
```

---

## Alpaca API Setup

### 1. Sign up for Alpaca

Register at [Alpaca](https://alpaca.markets/) to create an account and get your API key and secret. The free paper trading environment is recommended for testing.

### 2. Configure your API keys

Set your credentials as environment variables to keep them secure:

```bash
export APCA_API_KEY_ID='your_api_key'
export APCA_API_SECRET_KEY='your_secret_key'
export APCA_API_BASE_URL='https://paper-api.alpaca.markets'  # Paper trading endpoint
```

> Alternatively, store credentials in a local config file or a secrets manager. **Never commit your keys to version control.**

### 3. Verify connectivity

Before running the bot, confirm you can connect and authenticate with Alpaca using a simple test script or the bot's initial connection routine.

---

## Running the Bot

1. Run the data loader and data processor
2. Run the ML model generators (classifier + regressor)
3. Run the main trading script:

```bash
python main.py
```

Logs are written to `bot.log`, and executed trades are recorded in `trade_log.csv`.

---


## Disclaimer

This project is for educational purposes only and does not constitute financial advice. Trading involves risk, and past performance is not indicative of future results. Use at your own risk, and always test thoroughly in a paper trading environment before deploying with real funds.
