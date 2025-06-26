import os
import pandas as pd
from datetime import datetime, timedelta
import time
import alpaca_trade_api as tradeapi
from src.keys.live_config import API_KEY, SECRET_KEY, BASE_URL

"""
Alpaca 1-Minute Historical Data Fetcher and Saver

This script uses the Alpaca Markets API to fetch 1-minute historical bar data for a list of stock symbols. The data is collected in 30-day chunks (to avoid API limitations), aggregated, and saved in Parquet format for each symbol.

Main Workflow:
--------------
1. Authenticates with Alpaca using API credentials (`live_config.py`).
2. Iterates over a list of stock symbols.
3. For each symbol:
    - Loops through the date range in 30-day intervals.
    - Fetches 1-minute bar data (`get_bars`) using IEX feed.
    - Aggregates and sorts the resulting DataFrame.
    - Saves the data to `data/dataraw/<symbol>.parquet`.

Key Features:
-------------
- Timeframe: 1-minute bars (`'1Min'`)
- Feed: IEX (for free data access)
- Range: Past 365 days from current date (customizable)
- Storage Format: Parquet files (efficient and easy to load later)
- Output Directory: `data/dataraw/`

Dependencies:
-------------
- pandas
- datetime
- alpaca-trade-api
- pyarrow or fastparquet (for Parquet saving)
- Custom config file: `src/keys/live_config.py` containing:
    - `API_KEY`
    - `SECRET_KEY`
    - `BASE_URL`
"""

api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def fetch_and_save_data(symbols, start_date, end_date, 
                        timeframe='1Min', 
                        save_dir="data/dataraw"):
    os.makedirs(save_dir, exist_ok=True)

    for symbol in symbols:
        print(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}...")
        df_all = []
        current = start_date

        while current < end_date:
            next_chunk = min(current + timedelta(days=30), end_date)
            start_str = current.replace(microsecond=0).isoformat() + "Z"
            end_str = next_chunk.replace(microsecond=0).isoformat() + "Z"

            bars = api.get_bars(symbol, timeframe, start=start_str, end=end_str, feed='iex').df
            if bars.empty:
                print(f"No data for {symbol} from {current.date()} to {next_chunk.date()}")
            else:
                df_all.append(bars)

            current = next_chunk
            time.sleep(1)

        if df_all:
            full_df = pd.concat(df_all).sort_index()
            full_df.to_parquet(os.path.join(save_dir, f"{symbol}.parquet"))
            print(f"Saved {symbol} with {len(full_df)} rows in {save_dir}")
        else:
            print(f"No data collected for {symbol}")

if __name__ == "__main__":
    symbols = ['AAPL', 'MSFT', 'SMCI', 'AMD', 'GOOGL', 'META', 'AMZN', 'INMD', 
               'NFLX', 'NVDA', 'TSLA', 'BABA', 'INTC', 'CSCO', 'QCOM', 'AVGO', 
               'TXN', 'MU']

    # For seen data (training data):
    seen_end = datetime.now()
    seen_start = seen_end - timedelta(days=365)
    fetch_and_save_data(symbols, seen_start, seen_end, save_dir="data/dataraw")

    # For unseen data (validation/test data):
    unseen_end = datetime(2024, 5, 31)   # day before training starts
    unseen_start = unseen_end - timedelta(days=365)
    fetch_and_save_data(symbols, unseen_start, unseen_end, save_dir="data/dataraw_unseen")

