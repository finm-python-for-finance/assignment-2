import pandas as pd

class Strategy():
    """
    Base class for strategies that holds the market data, tickers, start and end dates, and a DataFrame for signals.
    Has function (meant to be overridden with your strategy logic) to generate signals, and a function to use those signals to backtest.
    """
    def __init__(self, market_data_df:pd.DataFrame, tickers:list[str]=None, start:str=None, end:str=None, name:str=None, **kwargs):
        self._market_data_df = market_data_df
        self._tickers = tickers
        if self._tickers is None:
            self._tickers = list(market_data_df.columns)

        self._start = start
        if self._start is None:
            self._start = str(market_data_df.index.min().date())

        self._end = end
        if self._end is None:
            self._end = str(market_data_df.index.max().date())

        self.name = name or self.__class__.__name__
        self.signals = None
        self.parameters = {}
        self.indicators = {}
        
        # Alias for compatibility with strategy classes
        self.prices = self._market_data_df

    def generate_signals(self) -> pd.DataFrame:
        """
        Output should be a DataFrame with a column for each ticker, values corresponding to +1 for buy, -1 for sell, 0 for no trade. Index same as market_data_df index.
        See this self.signals for an example.
        """

        self.signals = pd.DataFrame(0, index=self._market_data_df.index, columns=self._tickers)

        return self.signals
    
    def set_parameters(self, **kwargs):
        """Set strategy parameters."""
        self.parameters.update(kwargs)
    
    def record_indicator(self, name: str, data: pd.DataFrame):
        """Record indicator data for analysis."""
        self.indicators[name] = data
    
    def backtest(self, initial_capital:float=1_000_000) -> pd.DataFrame:
        """
        Backtests the strategy using the generated signals and returns:
        1) A dataframe with column for each ticker with the position over time (number of shares held) 
        2) A dataframe with a column for cash over time, a column for equity value over time, and a column for total portfolio value over time. 
            (AS OF START OF DAY, BEFORE ANY TRADES)
        
        DOES NOT ALLOW SHORTING (as stated in the assignment instructions).
        DOESNT ALLOW BUYING WHEN DON'T HAVE ENOUGH CASH.
        DOESNT ALLOW ORDERS OVER 1 SHARE.
        """

        if self.signals is None:
            self.generate_signals()

        position_df = pd.DataFrame(0, index=self._market_data_df.index, columns=self._tickers)

        cash_values = [initial_capital]
        equity_values = [0]

        # participation rate tracking
        cumulative_abs_returns = 0.0
        trade_count = 0
        stop_trading = False

        # Yes, I know this is slow, but it's the simplest way to get exactly the conditional execution logic we're asked for.
        for t in range(1, len(self._market_data_df)):
            date = self._market_data_df.index[t]
            prev_date = self._market_data_df.index[t-1]
            current_cash = cash_values[-1]
            current_positions = position_df.loc[prev_date].to_dict()

            for ticker in self._tickers:
                ticker_lastday_EOD_signal = self.signals.loc[prev_date, ticker]
                ticker_lastday_EOD_price = self._market_data_df.loc[prev_date, ticker]
                if ticker_lastday_EOD_signal == 1: # Buy one share
                    if current_cash >= ticker_lastday_EOD_price:
                        current_cash -= ticker_lastday_EOD_price
                        position_df.loc[date, ticker] = current_positions[ticker] + 1
                    else:
                        position_df.loc[date, ticker] = current_positions[ticker] # Can't afford to buy
                        print(f"Strategy {self.__class__.__name__} wanted to buy {ticker} on {prev_date.date()} at {ticker_lastday_EOD_price} but only had {current_cash} cash.")
                elif ticker_lastday_EOD_signal == -1: # Sell one share
                    if current_positions[ticker] >= 1:
                        current_cash += ticker_lastday_EOD_price
                        position_df.loc[date, ticker] = current_positions[ticker] - 1
                    else:
                        position_df.loc[date, ticker] = current_positions[ticker] # Can't sell what we don't have
                        print(f"Strategy {self.__class__.__name__} wanted to sell {ticker} on {prev_date.date()} at {ticker_lastday_EOD_price} but only had {current_positions[ticker]} shares.")
                else:
                    if ticker_lastday_EOD_signal != 0:
                        print(f"Strategy {self.__class__.__name__} generated invalid signal {ticker_lastday_EOD_signal} for {ticker} on {prev_date.date()}. Only -1, 0, +1 allowed.")
                    position_df.loc[date, ticker] = current_positions[ticker]
            
            #check participation threshold after each day trades
            if trade_count > 0 and (cumulative_abs_returns / trade_count) > 0.10:
                stop_trading = True

            cash_values.append(current_cash)
            equity_value = sum(position_df.loc[date, ticker] * self._market_data_df.loc[date, ticker] for ticker in self._tickers)
            equity_values.append(equity_value)

        money_df = pd.DataFrame({
            'Cash': cash_values,
            'Equity': equity_values
        }, index=self._market_data_df.index)
        money_df['Total'] = money_df['Cash'] + money_df['Equity']

        return position_df, money_df

class BenchmarkStrategy(Strategy):
    """
    A simple benchmark strategy that buys one of each ticker at the first day and holds until the end.
    """
    def generate_signals(self) -> pd.Series:
        signals = pd.DataFrame(0, index=self._market_data_df.index, columns=self._tickers)
        first_date = self._market_data_df.index.min()
        signals.loc[first_date] = 1 # Buy one of each ticker on the first day

        self.signals = signals

        return signals