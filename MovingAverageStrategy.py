from __future__ import annotations
from typing import Optional
import pandas as pd
from BenchmarkStrategy import Strategy

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
        signals = pd.DataFrame(0, index = self.prices.index, columns = self.prices.columns)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        self.signals = signals
        return signals

__all__ = ["MovingAverageStrategy"]
