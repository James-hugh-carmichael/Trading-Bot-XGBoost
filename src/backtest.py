import pandas as pd
import numpy as np
import joblib

df = pd.read_parquet("data/unseen_dataprocessed/training_data.parquet")

# Load models
clf_model = joblib.load("models/best_xgb_classifier.pkl")
reg_model = joblib.load("models/best_xgb_regressor.pkl")

# Parameters
clf_threshold = 0.58
reg_threshold = 0.002
STOP_LOSS_PCT = 0.06
TAKE_PROFIT_PCT = 0.1
INITIAL_CAPITAL = 100 
POSITION_SIZE_PCT = 0.2  

features = [col for col in df.columns if col not in ['target', 'target_return', 'future_close', 'symbol', 'index']]
X = df[features]
prices = df['close'].values

clf_probs = clf_model.predict_proba(X)[:, 1]
reg_preds = reg_model.predict(X)

capital = INITIAL_CAPITAL
positions = []
trades = []

for i in range(len(df)):
    price = prices[i]
    prob = clf_probs[i]
    pred_return = reg_preds[i]

    still_open_positions = []
    for position in positions:
        entry_price = position["entry_price"]
        direction = position["direction"]
        entry_capital = position["capital_allocated"]

        # Calculate return 
        if direction == "long":
            change = (price - entry_price) / entry_price
        else:  # short
            change = (entry_price - price) / entry_price

        exit_trade = False
        reason = None

        if direction == "long":
            if change <= -STOP_LOSS_PCT:
                reason = "stop_loss"
                exit_trade = True
            elif change >= TAKE_PROFIT_PCT:
                reason = "take_profit"
                exit_trade = True
        else:
            if change >= STOP_LOSS_PCT:
                reason = "stop_loss"
                exit_trade = True
            elif change <= -TAKE_PROFIT_PCT:
                reason = "take_profit"
                exit_trade = True

        # Signal reverse exit)
        if not exit_trade:
            if direction == "long" and (prob < (1 - clf_threshold) and pred_return < -reg_threshold):
                reason = "signal_reverse"
                exit_trade = True
            elif direction == "short" and (prob > clf_threshold and pred_return > reg_threshold):
                reason = "signal_reverse"
                exit_trade = True

        if exit_trade:
            # Calculate profit/loss on capital
            profit_loss = entry_capital * change
            capital += entry_capital + profit_loss  # Return capital + P/L
            trades.append({
                "entry_index": position["entry_index"],
                "exit_index": i,
                "entry_price": entry_price,
                "exit_price": price,
                "direction": direction,
                "return": change,
                "capital_allocated": entry_capital,
                "profit_loss": profit_loss,
                "reason": reason
            })
        else:
            still_open_positions.append(position)

    positions = still_open_positions

    if prob > clf_threshold and pred_return > reg_threshold:
        capital_to_use = capital * POSITION_SIZE_PCT
        if capital_to_use > 0:  # can buy at least one unit
            capital -= capital_to_use
            positions.append({
                "entry_index": i,
                "entry_price": price,
                "direction": "long",
                "capital_allocated": capital_to_use
            })

    elif prob < 0.25 and pred_return < -0.0065:
        capital_to_use = capital * POSITION_SIZE_PCT
        if capital_to_use > 0:
            capital -= capital_to_use
            positions.append({
                "entry_index": i,
                "entry_price": price,
                "direction": "short",
                "capital_allocated": capital_to_use
            })

# Close any remaining positions at last price
final_price = prices[-1]
for position in positions:
    direction = position["direction"]
    entry_price = position["entry_price"]
    entry_capital = position["capital_allocated"]
    ret = (final_price - entry_price) / entry_price if direction == "long" else (entry_price - final_price) / entry_price
    profit_loss = entry_capital * ret
    capital += entry_capital + profit_loss
    trades.append({
        "entry_index": position["entry_index"],
        "exit_index": len(df) - 1,
        "entry_price": entry_price,
        "exit_price": final_price,
        "direction": direction,
        "return": ret,
        "capital_allocated": entry_capital,
        "profit_loss": profit_loss,
        "reason": "end_of_data"
    })

# Convert trades to DataFrame
trades_df = pd.DataFrame(trades)
print(f"Classifier prob min: {clf_probs.min()}, max: {clf_probs.max()}")
print(f"Regressor pred min: {reg_preds.min()}, max: {reg_preds.max()}")

print(f"Total trades: {len(trades_df)}")
print(f"Win rate: {(trades_df['return'] > 0).mean():.2%}")
print(f"Average return per trade: {trades_df['return'].mean():.4f}")
print(f"Final capital: ${capital:.2f}")
print(f"Total return: {(capital / INITIAL_CAPITAL - 1):.2%}")

# Plot equity curve based on cumulative capital after each trade
trades_df = trades_df.sort_values('exit_index').reset_index(drop=True)
trades_df['cumulative_capital'] = INITIAL_CAPITAL + trades_df['profit_loss'].cumsum()
trades_df['cumulative_capital'].plot(title="Backtest Capital Curve")
