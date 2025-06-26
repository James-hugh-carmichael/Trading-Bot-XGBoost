import os
import pandas as pd
from datetime import datetime, timedelta
import time
import alpaca_trade_api as tradeapi
from src.keys.live_config import API_KEY, SECRET_KEY, BASE_URL

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def fetch_and_save_data(symbols, start_date, end_date, timeframe='1Min'):
    for symbol in symbols:
        print(f"Fetching {symbol}...")
        df_all = []

        current = start_date
        
        while current < end_date:
            next_month = current + timedelta(days=30)
            start_str = current.replace(microsecond=0).isoformat() + "Z"
            end_str = next_month.replace(microsecond=0).isoformat() + "Z"
            bars = api.get_bars(symbol, timeframe, start=start_str, end=end_str, feed='iex').df
            if bars.empty:
                print(f"No data for {symbol} from {current} to {next_month}")
            else:
                df_all.append(bars)
            current = next_month

        full_df = pd.concat(df_all).sort_index()
        full_df.to_parquet(f"data/dataraw/{symbol}.parquet")
        print(f"Saved {symbol} with {len(full_df)} rows")

        time.sleep(1)

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    symbols = ['AAPL', 'MSFT', 'SMCI', 'AMD', 'GOOGL', 'META', 'AMZN', 'INMD', 'NFLX', 'NVDA', 'TSLA', 'BABA', 'INTC', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'MU']
    end = datetime.now()
    start = end - timedelta(days=365) 
    fetch_and_save_data(symbols, start, end)
