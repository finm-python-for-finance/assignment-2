from datetime import datetime
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MarketDataPoint:
    timestamp: datetime
    symbol: str
    price: float

class Order:
    def __init__(self, symbol: str, quantity: int, price: float, status: str) -> None:
        self.symbol = symbol
        self.quantity = int(quantity)
        self.price = float(price)
        self.status = status

    def validate(self) -> None:
        if self.quantity == 0.0:
            raise OrderError(f"Invalid quantity: {self.quantity}")
        if self.price <= 0.0:
            raise OrderError(f"Invalid price: {self.price}")

    def update_status(self, new_status: str) -> None:
        self.status = new_status

class OrderError(Exception):
    """Raised when an order is invalid or cannot be created."""

class ExecutionError(Exception):
    """Raised when an order fails during simulated execution."""