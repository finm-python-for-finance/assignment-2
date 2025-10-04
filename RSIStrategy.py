from __future__ import annotations
from typing import Optional
import pandas as pd
from BenchmarkStrategy import Strategy

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
        signals = pd.DataFrame(0, index = self.prices.index, columns = self.prices.columns)
        signals[rsi < self.threshold] = 1
        signals[rsi > (100 - self.threshold)] = -1
        self.signals = signals
        return signals

__all__ = ["RSIStrategy"]
