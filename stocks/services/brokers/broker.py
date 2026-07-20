"""Abstract broker interface for placing orders."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Broker(ABC):
    """Broker that can buy and sell instruments."""

    @abstractmethod
    def buy(self, symbol: str, quantity: float) -> None:
        """Buy ``quantity`` of ``symbol``."""

    @abstractmethod
    def sell(self, symbol: str, quantity: float) -> None:
        """Sell ``quantity`` of ``symbol``."""
