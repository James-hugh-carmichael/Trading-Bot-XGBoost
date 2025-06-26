> Trading Bot with XGBoost Classifier & Regressor

This repository contains a Python-based trading bot that uses machine learning models (XGBoost classifier and regressor) to predict stock price movements and execute trades based on these predictions.

> Features
- Feature engineering with technical indicators and rolling statistics
- Classification and regression models trained on historical stock data
- Trade signal generation for long and short positions
- Performance evaluation with cumulative returns and win rate
- Visualization of model predictions and feature importance

> Getting Started

**Requirements**
- Python 3.8+
- pandas
- numpy
- matplotlib
- xgboost
- ta (technical analysis library)
- joblib
- yfinance
- alpaca-trade-api

Install dependencies with:
pip install -r requirements.txt

> Alpaca API Setup

Sign up for Alpaca
Register at Alpaca to create an account and get your API key and secret. You can use the free paper trading environment for testing.

Configure your API keys
Set your API credentials as environment variables to keep them secure:

export APCA_API_KEY_ID='your_api_key'
export APCA_API_SECRET_KEY='your_secret_key'
export APCA_API_BASE_URL='https://paper-api.alpaca.markets' # For paper trading

Alternatively, you can store them in a local config file or use a secrets manager. Do not commit your keys to version control.

Verify connectivity
Before running the bot, ensure you can connect and authenticate with Alpaca via a simple test script or your trading bot's initial connection.

> Running the Bot
Run the data loader and data processor, then run the ML model generators. 

Run the trading bot main script:
python main.py
Logs will be written to bot.log and executed trades will be recorded in trade_log.csv.

> Repository Structure
- src/             # Source code for bot, strategy, and API interactions
- models/          # Saved ML models (XGBoost classifier and regressor)
- data/            # Training and historical data
- trade_log.csv    # CSV file where trades are logged (generated at runtime)
- bot.log          # Runtime logs

> Security Notice

Never commit your API keys or any sensitive credentials to GitHub.

Use .gitignore to exclude any files or folders containing secrets or credentials.

Recommended .gitignore entries:

- *.env
- __pycache__/
- *.pyc
- bot.log
