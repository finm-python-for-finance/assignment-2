from __future__ import annotations

from typing import Optional

import pandas as pd

from base_strategy import Strategy


class RSIStrategy(Strategy):
    def __init__(
        self,
        prices: pd.DataFrame,
        lookback: int = 14,
        threshold: float = 30.0,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(prices, name=name or "RSIStrategy", **kwargs)
        if lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        self.lookback = int(lookback)
        self.threshold = float(threshold)
        self.set_parameters(lookback=self.lookback, threshold=self.threshold)

    def generate_signals(self) -> pd.DataFrame:
        price_diff = self.prices.diff()
        gains = price_diff.clip(lower=0.0)
        losses = -price_diff.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / self.lookback, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / self.lookback, adjust=False).mean()
        avg_loss = avg_loss.where(avg_loss > 0, 1e-9)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.clip(lower=0, upper=100)
        self.record_indicator("rsi", rsi)
        signals = (rsi < self.threshold).astype(float)
        return signals


__all__ = ["RSIStrategy"]



# MACD

from __future__ import annotations

from typing import Optional

import pandas as pd

from base_strategy import Strategy


class MACDStrategy(Strategy):
    def __init__(
        self,
        prices: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")
        super().__init__(prices, name=name or "MACDStrategy", **kwargs)
        self.fast_period = int(fast_period)
        self.slow_period = int(slow_period)
        self.signal_period = int(signal_period)
        self.set_parameters(
            fast_period=self.fast_period,
            slow_period=self.slow_period,
            signal_period=self.signal_period,
        )

    def generate_signals(self) -> pd.DataFrame:
        ema_fast = self.prices.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = self.prices.ewm(span=self.slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        crossover = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        signals = crossover.astype(float)
        self.record_indicator("macd_line", macd_line)
        self.record_indicator("signal_line", signal_line)
        return signals


__all__ = ["MACDStrategy"]

# Moving Average
from __future__ import annotations

from typing import Optional

import pandas as pd

from base_strategy import Strategy


class MovingAverageStrategy(Strategy):
    def __init__(
        self,
        prices: pd.DataFrame,
        short_window: int = 20,
        long_window: int = 50,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        super().__init__(prices, name=name or "MovingAverageStrategy", **kwargs)
        self.short_window = int(short_window)
        self.long_window = int(long_window)
        self.set_parameters(short_window=self.short_window, long_window=self.long_window)

    def generate_signals(self) -> pd.DataFrame:
        short_ma = self.prices.rolling(window=self.short_window, min_periods=self.short_window).mean()
        long_ma = self.prices.rolling(window=self.long_window, min_periods=self.long_window).mean()
        self.record_indicator("short_ma", short_ma)
        self.record_indicator("long_ma", long_ma)
        signals = (short_ma > long_ma).astype(float)
        return signals

__all__ = ["MovingAverageStrategy"]

# benchmark
from __future__ import annotations

from typing import Optional

import pandas as pd

from base_strategy import Strategy


class BenchmarkStrategy(Strategy):
    """Buy a capped number of shares for every ticker on the first session."""

    def __init__(
        self,
        prices: pd.DataFrame,
        volume: pd.DataFrame,
        initial_cash: float = 1_000_000.0,
        participation: float = 0.05,
        name: Optional[str] = None,
    ) -> None:
        super().__init__(prices, initial_cash=initial_cash, signal_lag=0, name=name or "BenchmarkStrategy")
        if volume is None or volume.empty:
            raise ValueError("Volume data is required for the benchmark strategy")
        self.volume = volume.reindex(self.prices.index).reindex(columns=self.prices.columns).fillna(0.0)
        self.participation = float(max(0.0, min(participation, 0.2)))
        self.set_parameters(participation=self.participation, initial_cash=initial_cash)

    def generate_signals(self) -> pd.DataFrame:
        shares = pd.DataFrame(0.0, index=self.prices.index, columns=self.prices.columns)
        first_session = self.prices.index[0]
        price_row = self.prices.loc[first_session]
        volume_row = self.volume.loc[first_session] if first_session in self.volume.index else pd.Series(0.0, index=self.prices.columns)

        per_ticker_budget = self.initial_cash / max(1, len(self.prices.columns))
        first_day_orders = pd.Series(0.0, index=self.prices.columns)

        for ticker in self.prices.columns:
            price = price_row[ticker]
            volume = volume_row.get(ticker, 0.0)
            if pd.isna(price) or price <= 0 or pd.isna(volume) or volume <= 0:
                continue

            max_by_cash = int(per_ticker_budget // price)
            max_by_volume = int(volume * self.participation)
            order_size = max(0, min(max_by_cash, max_by_volume))
            if order_size > 0:
                first_day_orders[ticker] = order_size

        shares.loc[first_session] = first_day_orders
        allocation_frame = pd.DataFrame([first_day_orders], index=[first_session])
        self.record_indicator("benchmark_allocation", allocation_frame)
        return shares

    def adjust_order(
        self,
        date: pd.Timestamp,
        ticker: str,
        requested_shares: int,
        price: float,
    ) -> int:
        if date not in self.volume.index:
            return requested_shares
        daily_volume = float(self.volume.at[date, ticker])
        if daily_volume <= 0:
            return 0
        cap = int(daily_volume * self.participation)
        return max(0, min(requested_shares, cap))


__all__ = ["BenchmarkStrategy"]

# Volatility Breakout
from __future__ import annotations

from typing import Optional

import pandas as pd

from base_strategy import Strategy


class VolatilityBreakoutStrategy(Strategy):
    def __init__(
        self,
        prices: pd.DataFrame,
        lookback: int = 20,
        name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(prices, name=name or "VolatilityBreakoutStrategy", **kwargs)
        self.lookback = int(lookback)
        if self.lookback <= 1:
            raise ValueError("lookback must be greater than 1")
        self.set_parameters(lookback=self.lookback)

    def generate_signals(self) -> pd.DataFrame:
        returns = self.prices.pct_change()
        rolling_std = returns.rolling(window=self.lookback, min_periods=self.lookback).std()
        self.record_indicator("returns", returns)
        self.record_indicator("rolling_std", rolling_std)
        signals = (returns > rolling_std).astype(float)
        return signals


__all__ = ["VolatilityBreakoutStrategy"]

# Base Strategy
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class Trade:
    date: pd.Timestamp
    ticker: str
    quantity: int
    price: float
    cash_before: float
    cash_after: float


@dataclass
class BacktestResult:
    name: str
    portfolio: pd.DataFrame
    positions: pd.DataFrame
    signals: pd.DataFrame
    execution_signals: pd.DataFrame
    trades: List[Trade]
    skipped_orders: List[Dict[str, object]]
    indicators: Dict[str, pd.DataFrame]
    parameters: Dict[str, object]


class Strategy(ABC):
    def __init__(
        self,
        prices: pd.DataFrame,
        initial_cash: float = 1_000_000.0,
        signal_lag: int = 1,
        name: Optional[str] = None,
    ) -> None:
        if prices.empty:
            raise ValueError("Price data must not be empty")
        self.prices = prices.sort_index()
        self.initial_cash = float(initial_cash)
        self.signal_lag = max(0, int(signal_lag))
        self.name = name or self.__class__.__name__
        self.indicators: Dict[str, pd.DataFrame] = {}
        self.parameters: Dict[str, object] = {}
        self._result: Optional[BacktestResult] = None

    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """Return a DataFrame of non-negative share targets per ticker."""

    def record_indicator(self, label: str, data: pd.DataFrame) -> None:
        self.indicators[label] = data.copy()

    def set_parameters(self, **kwargs: object) -> None:
        self.parameters.update(kwargs)

    def adjust_order(
        self,
        date: pd.Timestamp,
        ticker: str,
        requested_shares: int,
        price: float,
    ) -> int:
        """Hook for subclasses to cap or amend orders before execution."""
        return requested_shares

    def run(self) -> BacktestResult:
        if self._result is not None:
            return self._result

        raw_signals = self.generate_signals()
        signals = self._prepare_signals(raw_signals)
        execution_signals = signals.shift(self.signal_lag, fill_value=0)

        tickers = list(self.prices.columns)
        cash = self.initial_cash
        positions = {ticker: 0 for ticker in tickers}
        trades: List[Trade] = []
        skipped_orders: List[Dict[str, object]] = []
        portfolio_records: List[Dict[str, object]] = []
        position_records: List[pd.Series] = []

        for date in self.prices.index:
            price_row = self.prices.loc[date]
            order_row = execution_signals.loc[date] if date in execution_signals.index else pd.Series(0, index=tickers)

            for ticker, signal_value in order_row.items():
                if signal_value <= 0:
                    continue

                price = price_row[ticker]
                if pd.isna(price) or price <= 0:
                    skipped_orders.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "invalid price",
                            "requested_shares": signal_value,
                        }
                    )
                    continue

                requested = int(signal_value)
                if requested <= 0:
                    continue

                adjusted = max(0, int(self.adjust_order(date, ticker, requested, float(price))))
                if adjusted == 0:
                    skipped_orders.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "order capped by strategy",
                            "requested_shares": requested,
                        }
                    )
                    continue

                affordable = int(cash // price)
                if affordable <= 0:
                    skipped_orders.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "insufficient cash",
                            "requested_shares": adjusted,
                        }
                    )
                    continue

                shares_to_buy = min(adjusted, affordable)
                if shares_to_buy <= 0:
                    skipped_orders.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "insufficient cash",
                            "requested_shares": adjusted,
                        }
                    )
                    continue

                cost = shares_to_buy * float(price)
                cash_before = cash
                cash -= cost
                positions[ticker] += shares_to_buy
                trades.append(
                    Trade(
                        date=date,
                        ticker=ticker,
                        quantity=shares_to_buy,
                        price=float(price),
                        cash_before=cash_before,
                        cash_after=cash,
                    )
                )

                if shares_to_buy < adjusted:
                    skipped_orders.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "reason": "partially filled (cash limit)",
                            "requested_shares": adjusted,
                            "filled_shares": shares_to_buy,
                        }
                    )

            position_series = pd.Series(
                [positions[t] for t in tickers], index=tickers, name=date, dtype=int
            )
            holdings_value = float((position_series * price_row).sum())
            total_value = cash + holdings_value
            portfolio_records.append(
                {
                    "Date": date,
                    "cash": cash,
                    "holdings_value": holdings_value,
                    "total_value": total_value,
                    "pnl": total_value - self.initial_cash,
                }
            )
            position_records.append(position_series)

        portfolio_df = pd.DataFrame(portfolio_records).set_index("Date").sort_index()
        positions_df = pd.DataFrame(position_records).sort_index()

        self._result = BacktestResult(
            name=self.name,
            portfolio=portfolio_df,
            positions=positions_df,
            signals=signals,
            execution_signals=execution_signals,
            trades=trades,
            skipped_orders=skipped_orders,
            indicators={key: value.copy() for key, value in self.indicators.items()},
            parameters={**self.parameters},
        )
        return self._result

    def _prepare_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        aligned = signals.reindex(self.prices.index).reindex(columns=self.prices.columns)
        aligned = aligned.fillna(0.0)
        return aligned.astype(float)


__all__ = ["Strategy", "Trade", "BacktestResult"]
