from stocks.services.brokers import Broker

class SaxoTrader(Broker):
    def __init__(self, token: str) -> None:
        self.token = token

    def buy(self, symbol: str, quantity: float) -> None:
        raise NotImplementedError

    def sell(self, symbol: str, quantity: float) -> None:
        raise NotImplementedError

