from __future__ import annotations
from typing import Optional
import pandas as pd
from BenchmarkStrategy import Strategy

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
        signals = pd.DataFrame(0, index = self.prices.index, columns = self.prices.columns)
        signals[returns > rolling_std] = 1
        signals[returns < -rolling_std] = -1
        self.signals = signals
        return signals


__all__ = ["VolatilityBreakoutStrategy"]