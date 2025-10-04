import pandas as pd
from pathlib import Path
import yfinance as yf
import os

class PriceLoader:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.available_tickers = []
        self.ticker_data = {}

    def get_spy_tickers(self):
        """
        Reads from a locally saved recent html file of the S&P 500 companies Wikipedia page.
        """
        tickers = pd.read_html('spywikipedia.html')[0]
        return tickers['Symbol'].tolist()
    
    def _load_one(self, ticker:list[str], start:str, end:str):
        """
        Loads one ticker from local parquet file, getting it from yfinance if not local.
        """
        ticker_file_name = f"{ticker}_{start}_{end}.parquet"
        if not os.path.exists(self.data_dir / ticker_file_name):
            print(f"Downloading {ticker} from yfinance from {start} to {end} (not found locally)")
            self._load_yfinance([ticker], start, end)

        return pd.read_parquet(self.data_dir / ticker_file_name)
    
    def _load_yfinance(self, tickers:list[str], start:str, end:str):
        """
        Loads and saves yfinance Adj Close from start to end for selected tickers.
        """
        df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for t in tickers:
            ticker_file_name = f"{t}_{start}_{end}.parquet"
            if not os.path.exists(self.data_dir / ticker_file_name):
                df["Adj Close"].to_parquet(self.data_dir / ticker_file_name)
                print(f"Saved {t} from {start} to {end} at {self.data_dir / ticker_file_name}")


    def load(self, tickers:list[str], start:str, end:str) -> dict[str, pd.Series]:
        """
        Loads Adj Close series for selected tickers from start to end and returns a dict of ticker to series.
        """
        print(f"Loading Tickers from {start} to {end}: {tickers}")
        if tickers is None:
            tickers = self.get_spy_tickers()
        ticker_series = {}
        for t in tickers:
            try:
                s = self._load_one(t, start, end)
                ticker_series[t] = s[t]
                self.available_tickers.append(t)
            except Exception as e:
                print(f"Error loading {t}: {e}")
        self.ticker_data = ticker_series
    
    def get_ticker_data(self, tickers:list[str]) -> pd.DataFrame:
        """
        Returns the loaded data for specific tickers time-aligned in a DataFrame.

        Drops any columns with NaN values since it wouldn't be fair to backtest them.
        """
        ticker_series = {t: self.ticker_data[t] for t in tickers if t in self.ticker_data}
        df = pd.DataFrame(ticker_series)
        df.dropna(axis=1, inplace=True)

        return df
