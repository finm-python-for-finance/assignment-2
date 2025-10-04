from __future__ import annotations
from typing import Optional
import pandas as pd
from BenchmarkStrategy import Strategy

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
        signals = pd.DataFrame(0, index = self.prices.index, columns = self.prices.columns)
        buy_signal = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        sell_signal = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        self.record_indicator("macd_line", macd_line)
        self.record_indicator("signal_line", signal_line)
        self.signals = signals
        return signals

__all__ = ["MACDStrategy"]
