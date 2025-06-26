import alpaca_trade_api as tradeapi
import pandas as pd
import yfinance as yf
from src.keys.paper_config import API_KEY, SECRET_KEY, BASE_URL
from datetime import datetime, timedelta
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from src.trading_strategy import build_features


api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

data_client = StockHistoricalDataClient(
    api_key=API_KEY,
    secret_key=SECRET_KEY,
    sandbox=False         
)

class User_Actions:
    def __init__(self,
                 api_key=API_KEY,
                 secret_key=SECRET_KEY,
                 base_url=BASE_URL):
        # Trading (paper) client
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.account = self.api.get_account()

    def create_watchlist(self, name, symbols):
        try:
            watchlist = self.api.create_watchlist(name=name, symbols=symbols)
            print(f"Watchlist '{name}' created with id: {watchlist.id}")
            return watchlist
        except Exception as e:
            print(f"Error creating watchlist '{name}': {e}")
            return None

    def get_account_info(self):
        return {
            "status": self.account.status,
            "equity": float(self.account.equity),
            "buying_power": float(self.account.buying_power)
        }

    def get_portfolio_change(self):
        if not self.account.last_equity:
            print("No previous equity value found.")
            return None
        return float(self.account.equity) - float(self.account.last_equity)

    def get_cash_balance(self):
        return float(self.account.cash)

    def get_positions(self):
        try:
            positions = self.api.list_positions()
            return [p.symbol for p in positions]
        except Exception as e:
            print(f"Error fetching positions: {e}")
            return []

    def get_prices(self, symbols):
        prices = {}
        if isinstance(symbols, str):
            symbols = [symbols]

        for symbol in symbols:
            try:
                df = yf.download(tickers=symbol, interval='1m', period='5d', progress=False, auto_adjust=True)
                if df.empty:
                    print(f"No price data found for {symbol} on yfinance.")
                    prices[symbol] = None
                else:
                    # Use last close price available
                    prices[symbol] = df['Close'].iloc[-1]
            except Exception as e:
                print(f"Error fetching price for {symbol} from yfinance: {e}")
                prices[symbol] = None

        return prices


    def get_intraday_yfinance(self, symbol, interval='1m', period='5d'):
        try:
            df = yf.download(tickers=symbol, interval=interval, period=period, progress=False, auto_adjust=True)
            if df.empty:
                print(f"No intraday data found for {symbol} (yfinance fallback)")
                return None

            df.index = df.index.tz_localize(None)

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [' '.join(col).strip() for col in df.columns]
            else:
                # just convert all columns to strings (in case they're not)
                df.columns = df.columns.astype(str)


            df.columns = [col.replace(f' {symbol}', '').lower() for col in df.columns]

            # Calculate features here
            df = build_features(df)

            return df
        except Exception as e:
            print(f"Error fetching yfinance data for {symbol}: {e}")
            return None


    def get_historical_data(self, symbol, timeframe='minute', limit=400):
        try:
            if timeframe == 'minute':
                days_needed = max(1, (limit // 390) + 1)
                return self.get_intraday_yfinance(symbol, interval='1m', period=f'{days_needed}d')
            else:
                return self.get_intraday_yfinance(symbol, interval='1d', period='250d')
        except Exception as e:
            print(f"Error fetching historical data for {symbol} via yfinance: {e}")
            return None

    def submit_order(self, symbol, qty, side, order_type='market', time_in_force='gtc'):
        try:
            return self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force
            )
        except Exception as e:
            print(f"Error submitting order for {symbol}: {e}")
            return None

    def close_position(self, symbol):
        try:
            self.api.close_position(symbol)
        except Exception as e:
            print(f"Error closing position for {symbol}: {e}")

    def close_all_positions(self):
        try:
            for pos in self.api.list_positions():
                self.api.close_position(pos.symbol)
        except Exception as e:
            print(f"Error closing all positions: {e}")

    def is_market_open(self):
        clock = self.api.get_clock()
        return clock.is_open

    def get_watchlist_symbols(self):
        return [
            'AAPL', 'MSFT', 'SMCI', 'AMD', 'GOOGL', 'META',
            'AMZN', 'INMD', 'NFLX', 'NVDA', 'TSLA', 'BABA',
            'INTC', 'CSCO', 'QCOM', 'AVGO', 'TXN', 'MU'
        ]

if __name__ == "__main__":
    user = User_Actions()
    print("Account:", user.get_account_info())
    print("Cash:", user.get_cash_balance())
    print("Positions:", user.get_positions())
    print("Prices:", user.get_prices(user.get_watchlist_symbols()[:5]))
    print("History:", user.get_historical_data("AAPL", timeframe='minute', limit=60).tail())
