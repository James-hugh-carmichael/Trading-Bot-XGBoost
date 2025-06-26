import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator

def build_features(df):
    # VWAP calculation
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()

    # Returns
    df['return_1m'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))

    # Rolling features
    windows = [5, 10, 30, 60, 120, 390]
    for w in windows:
        df[f'close_mean_{w}'] = df['close'].rolling(window=w).mean()
        df[f'close_std_{w}'] = df['close'].rolling(window=w).std()
        df[f'return_{w}'] = df['close'].pct_change(periods=w)

    # Lag features
    for lag in range(1, 6):
        df[f'lag_close_{lag}'] = df['close'].shift(lag)

    # Technical indicators
    df['rsi'] = RSIIndicator(df['close']).rsi()
    macd_obj = MACD(df['close'])
    df['macd'] = macd_obj.macd()
    df['macd_signal'] = macd_obj.macd_signal()
    bb = BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    # Time features
    df['minute'] = df.index.minute
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Trade count proxy to match classifier training
    df['trade_count'] = df['close'].diff().ne(0).astype(int).rolling(window=5).sum()

    return df
