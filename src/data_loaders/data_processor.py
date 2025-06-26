import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

"""
Stock Data Preprocessing Pipeline for Machine Learning (Seen & Unseen Data)

This script transforms raw, minute-level stock price data into a feature-rich dataset suitable for use in machine learning models. It supports processing both **training (seen)** and **prediction (unseen)** data

Key Functionality:
------------------
- Reads raw `.parquet` files 
- Outputs a processed `.parquet` file to the desired output path. (will need to be specified in the script)
- Automatically generates technical indicators, statistical features, and classification/regression targets.

Features Engineered:
--------------------
1. **Return Features**:
   - 1-minute percent return (`return_1m`)
   - 1-minute log return (`log_return`)
   - Rolling returns and statistics (mean, std) over windows of 5–390 minutes

2. **Lag Features**:
   - Previous close prices for 1 to 5 minutes (`lag_close_1` to `lag_close_5`)

3. **Technical Indicators**:
   - RSI, MACD
   - Bollinger Band high and low values
   - On-Balance Volume (OBV)

4. **Time-Based Features**:
   - Minute of hour, hour of day, and day of week

5. **Targets**:
   - `target`: Binary label (1 if return over `LOOKAHEAD_MINUTES` > `THRESHOLD`)
   - `target_return`: Actual future return over that period

Constants:
----------
- `LOOKAHEAD_MINUTES = 60`: Horizon for target calculation.
- `THRESHOLD = 0.002`: Minimum future return required to classify as a positive target.
- `RAW_DATA_DIR 
- `PROCESSED_DATA_DIR 
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

LOOKAHEAD_MINUTES = 60  
THRESHOLD = 0.002 

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


def process_data(raw_data_dir, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    all_dfs = []
    for filename in tqdm(os.listdir(raw_data_dir)):
        if filename.endswith(".parquet"):
            symbol = filename.replace(".parquet", "")
            path = os.path.join(raw_data_dir, filename)
            df = process_symbol(symbol, path)
            all_dfs.append(df)

    full_df = pd.concat(all_dfs)
    full_df.to_parquet(output_file)
    print(f"Saved processed data to {output_file} with {len(full_df)} rows.")


if __name__ == "__main__":
    # Example usage:
    # For unseen data
    process_data(
        raw_data_dir="data/dataraw_unseen",
        output_file="data/unseen_dataprocessed/training_data.parquet"
    )
    
    # For seen data
    process_data(
        raw_data_dir="data/dataraw",
        output_file="data/dataprocessed/training_data.parquet"
    )
