import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

RAW_DATA_DIR = "data/dataraw_unseen"
PROCESSED_DATA_DIR = "data/unseen_dataprocessed"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

LOOKAHEAD_MINUTES = 60  
THRESHOLD = 0.002 

OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "training_data.parquet")

def process_symbol(symbol, path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df[df['volume'] > 0]  # Remove no-trade periods

    # Returns
    df['return_1m'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Rolling windows
    for w in [5, 10, 30, 60, 120, 390]:
        df[f'close_mean_{w}'] = df['close'].rolling(w).mean()
        df[f'close_std_{w}'] = df['close'].rolling(w).std()
        df[f'return_{w}'] = df['close'].pct_change(w)

    # Lag features
    for lag in range(1, 6):
        df[f'lag_close_{lag}'] = df['close'].shift(lag)

    # Technical indicators
    df['rsi'] = RSIIndicator(df['close']).rsi()
    df['macd'] = MACD(df['close']).macd()
    bb = BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()

    # Time-based features
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Target: binary classification + regression return
    df['future_close'] = df['close'].shift(-LOOKAHEAD_MINUTES)
    df['target'] = ((df['future_close'] - df['close']) / df['close'] > THRESHOLD).astype(int)
    df['target_return'] = df['future_close'] / df['close'] - 1

    df = df.dropna()
    df['symbol'] = symbol
    return df


def main():
    all_dfs = []
    for filename in tqdm(os.listdir(RAW_DATA_DIR)):
        if filename.endswith(".parquet"):
            symbol = filename.replace(".parquet", "")
            path = os.path.join(RAW_DATA_DIR, filename)
            df = process_symbol(symbol, path)
            all_dfs.append(df)

    full_df = pd.concat(all_dfs)
    full_df.to_parquet(OUTPUT_FILE)
    print(f"Saved processed training data to {OUTPUT_FILE} with {len(full_df)} rows.")


if __name__ == "__main__":
    main()
