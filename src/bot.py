import time
from datetime import datetime, timedelta
import pandas as pd
import logging
from src.alpaca_api import User_Actions
from src.trading_strategy import build_features

"""
Live Trading Bot for Alpaca Using Classifier + Regressor Strategy

This script executes a fully automated stock trading bot integrated with Alpaca's API.
It uses machine learning models (a classifier and a regressor) to decide when to enter 
and exit long/short positions based on real-time 1-minute stock data.

Key Features:
- **Model-based Strategy**:
    - **Long Entry**: Classifier probability > 0.58 AND Regressor return > 0.002
    - **Short Entry**: Classifier probability < 0.25 AND Regressor return < -0.0065
- **Risk Management**:
    - Stop Loss: 6%
    - Take Profit: 10%
    - Position Sizing: 10% of available cash per trade
    - Maximum number of simultaneous trades (default: 1000)
    - Optional cooldown period after exiting a trade
- **Logging & Monitoring**:
    - Trades logged to a CSV file (`trade_log.csv`)
    - Bot activity logged to a file (`bot.log`)
    - Errors caught and logged with retry after delay
- **Data Pipeline**:
    - Live prices, positions, and historical OHLCV data fetched from Alpaca
    - Feature engineering using the same `build_features` method from training
    - Real-time predictions using loaded XGBoost models
- **Structure**:
    - Entry and exit logic executed in a continuous loop (every 60 seconds)
    - Ensures market is open before executing
    - Handles exceptions gracefully to maintain uptime

This script is designed for running in a production or paper trading environment 
and assumes the presence of:
- `User_Actions` class for interacting with Alpaca API
- Pre-trained classifier and regressor models
- A feature builder function (`build_features`) compatible with your model inputs
"""

actions = User_Actions()

# --- Config ---
STOP_LOSS_PCT = 0.06
TAKE_PROFIT_PCT = 0.1
TRADE_LOG_PATH = "trade_log.csv"
COOLDOWN_MINUTES = 0
MAX_TRADES = 1000


active_trades = {} 
cooldown_tracker = {}

# Logging Setup 
logging.basicConfig(level=logging.INFO, filename="bot.log", filemode="a",
                    format="%(asctime)s - %(levelname)s - %(message)s")

# log trades 
def log_trade(symbol, action, price, qty, reason, direction):
    record = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "price": price,
        "qty": qty,
        "reason": reason,
        "direction": direction
    }
    df = pd.DataFrame([record])
    if not pd.io.common.file_exists(TRADE_LOG_PATH):
        df.to_csv(TRADE_LOG_PATH, index=False)
    else:
        df.to_csv(TRADE_LOG_PATH, mode='a', index=False, header=False)

# Bot
def run_trading_bot(classifier, regressor, clf_threshold=0.58, reg_threshold=0.002):
    while True:
        try:
            if not actions.is_market_open():
                print("Market closed, sleeping")
                logging.info("Market closed. Sleeping...")
                time.sleep(60)
                continue

            watchlist = actions.get_watchlist_symbols()
            prices = actions.get_prices(watchlist)
            raw_positions = actions.get_positions()
            position_symbols = set(raw_positions)
            cash = float(actions.get_cash_balance())

            print(f"Watchlist: {watchlist}, Cash: {cash}")

            for symbol in watchlist:
                now = datetime.now()

                # Skip if cooling down
                if symbol in cooldown_tracker and now < cooldown_tracker[symbol]:
                    print(f"{symbol} is cooling down")
                    continue

                # Limit active trades
                if len(active_trades) >= MAX_TRADES and symbol not in active_trades:
                    print(f"Max trades reached, skipping {symbol}")
                    continue

                df = actions.get_historical_data(symbol, timeframe='1Min', limit=400)
                if df is None or df.empty:
                    print(f"No historical data for {symbol}")
                    continue
                df = build_features(df)
            
                if df.empty:
                    print(f"Features build returned empty DataFrame for {symbol}")
                    continue

                latest = df.iloc[[-1]]  #
                expected_columns = classifier.get_booster().feature_names
                X = latest[expected_columns]

                prob = classifier.predict_proba(X)[0][1]

                pred_return = regressor.predict(X)[0]
                price = prices.get(symbol)

                trade = active_trades.get(symbol)
                is_active = symbol in position_symbols and trade is not None
                
                
                # --- Entry Logic ---
                if not is_active:
                    try:
                        current_price = price.iloc[0]
                    except Exception as e:
                        logging.warning(f"Failed to extract price for {symbol}: {e}")
                        continue


                    qty = int((cash * 0.1) // current_price)  # 10% of cash per trade
                    if qty <= 0:
                        continue

                    # Long entry
                    if prob > clf_threshold and pred_return > reg_threshold:
                        actions.submit_order(symbol, qty, side='buy')
                        active_trades[symbol] = {
                            "entry_price": price,
                            "qty": qty,
                            "direction": "long",
                            "entry_time": now
                        }
                        log_trade(symbol, "buy", price, qty, reason="long_entry", direction="long")
                        logging.info(f"LONG: Bought {qty} {symbol} at {price:.2f}")

                    # Short entry
                    elif prob < 0.25 and pred_return <  -0.0065:
                        actions.submit_order(symbol, qty, side='sell')
                        active_trades[symbol] = {
                            "entry_price": price,
                            "qty": qty,
                            "direction": "short",
                            "entry_time": now
                        }
                        log_trade(symbol, "sell", price, qty, reason="short_entry", direction="short")
                        logging.info(f"SHORT: Sold {qty} {symbol} at {price:.2f}")

                # --- Exit Logic ---
                elif is_active:
                    entry_price = trade["entry_price"]
                    direction = trade["direction"]
                    qty = trade["qty"]
                    change = (price - entry_price) / entry_price

                    if direction == "long":
                        if change <= -STOP_LOSS_PCT:
                            reason = "long_stop_loss"
                        elif change >= TAKE_PROFIT_PCT:
                            reason = "long_take_profit"
                        else:
                            continue
                    elif direction == "short":
                        if change >= STOP_LOSS_PCT:
                            reason = "short_stop_loss"
                        elif change <= -TAKE_PROFIT_PCT:
                            reason = "short_take_profit"
                        else:
                            continue

                    actions.close_position(symbol)
                    log_trade(symbol, "close", price, qty, reason=reason, direction=direction)
                    logging.info(f"{reason.upper()} on {symbol} at {price:.2f}")
                    active_trades.pop(symbol, None)
                    cooldown_tracker[symbol] = now + timedelta(minutes=COOLDOWN_MINUTES)

            time.sleep(60)

        except Exception as e:
            logging.exception(f"Unexpected error in trading loop: {e}")
            time.sleep(60)  # Prevent crash loop
