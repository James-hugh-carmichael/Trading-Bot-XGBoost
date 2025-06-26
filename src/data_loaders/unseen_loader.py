import os
import pandas as pd
from datetime import datetime, timedelta
import time
import alpaca_trade_api as tradeapi
from src.keys.live_config import API_KEY, SECRET_KEY, BASE_URL

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def fetch_and_save_unseen_data(symbols, start_date, end_date, timeframe='1Min'):
    os.makedirs("data/dataraw_unseen", exist_ok=True)

    for symbol in symbols:
        print(f"Fetching unseen data for {symbol}...")
        df_all = []
        current = start_date

        while current < end_date:
            next_chunk = min(current + timedelta(days=30), end_date)
            start_str = current.replace(microsecond=0).isoformat() + "Z"
            end_str = next_chunk.replace(microsecond=0).isoformat() + "Z"

            bars = api.get_bars(symbol, timeframe, start=start_str, end=end_str, feed='iex').df
            if bars.empty:
                print(f"No data for {symbol} from {current} to {next_chunk}")
            else:
                df_all.append(bars)

            current = next_chunk
            time.sleep(1)

        if df_all:
            full_df = pd.concat(df_all).sort_index()
            full_df.to_parquet(f"data/dataraw_unseen/{symbol}.parquet")
            print(f"Saved unseen data for {symbol} with {len(full_df)} rows")
        else:
            print(f"No data collected for {symbol}")

if __name__ == "__main__":
    # Adjust this to match the end date of your training data
    end = datetime(2024, 5, 31)   # Day before training starts
    start = end - timedelta(days=365)  # One year of unseen data

    symbols = ['AAPL', 'MSFT', 'SMCI', 'AMD', 'GOOGL', 'META', 'AMZN', 'INMD', 'NFLX', 'NVDA', 'TSLA', 'BABA', 'INTC', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'MU']

    fetch_and_save_unseen_data(symbols, start, end)
