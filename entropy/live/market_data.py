"""Real-time market data — quotes, bars, tick subscriptions.

Provides a thin wrapper over ib_insync's market-data methods with
structured logging and a local cache of latest snapshots.
"""

from __future__ import annotations

import datetime as dt
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger


class MarketDataManager:
    """Subscribe to and query real-time market data from IB.

    Usage::

        md = MarketDataManager(gateway)
        md.subscribe(["AAPL", "MSFT"])
        snap = md.snapshot("AAPL")      # {bid, ask, last, volume, ...}
        md.unsubscribe_all()
    """

    def __init__(self, gateway):
        """
        Parameters
        ----------
        gateway : IBGateway
            Connected gateway instance.
        """
        self._gw = gateway
        self._contracts: Dict[str, object] = {}   # ticker → Contract
        self._tickers: Dict[str, object] = {}      # ticker → ib_insync.Ticker
        self._callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # Contract helpers
    # ------------------------------------------------------------------

    def _make_contract(self, symbol: str, sec_type: str = "STK",
                       exchange: str = "SMART", currency: str = "USD"):
        """Create and qualify an IB contract."""
        from ib_insync import Stock
        contract = Stock(symbol, exchange, currency)
        ib = self._gw.ib
        qualified = ib.qualifyContracts(contract)
        if qualified:
            self._contracts[symbol] = qualified[0]
            return qualified[0]
        logger.warning("Could not qualify contract for {}", symbol)
        return contract

    def get_contract(self, symbol: str):
        if symbol not in self._contracts:
            self._make_contract(symbol)
        return self._contracts.get(symbol)

    # ------------------------------------------------------------------
    # Streaming subscriptions
    # ------------------------------------------------------------------

    def subscribe(self, symbols: List[str]) -> None:
        """Request streaming market data for a list of symbols."""
        ib = self._gw.ib
        for sym in symbols:
            if sym in self._tickers:
                continue
            contract = self.get_contract(sym)
            if contract is None:
                continue
            ticker = ib.reqMktData(contract, genericTickList="",
                                   snapshot=False, regulatorySnapshot=False)
            self._tickers[sym] = ticker
            logger.info("Subscribed to market data: {}", sym)

    def unsubscribe(self, symbol: str) -> None:
        """Cancel market data for one symbol."""
        if symbol in self._tickers:
            ib = self._gw.ib
            contract = self.get_contract(symbol)
            if contract:
                ib.cancelMktData(contract)
            del self._tickers[symbol]
            logger.info("Unsubscribed from market data: {}", symbol)

    def unsubscribe_all(self) -> None:
        """Cancel all active market data subscriptions."""
        for sym in list(self._tickers.keys()):
            self.unsubscribe(sym)

    # ------------------------------------------------------------------
    # Snapshot queries
    # ------------------------------------------------------------------

    def snapshot(self, symbol: str) -> Dict[str, object]:
        """Return the latest quote snapshot for *symbol*.

        Returns dict with keys: bid, ask, last, mid, volume, high, low, close.
        """
        ticker = self._tickers.get(symbol)
        if ticker is None:
            return {}

        bid = _safe_float(ticker.bid)
        ask = _safe_float(ticker.ask)
        last = _safe_float(ticker.last)
        mid = (bid + ask) / 2 if bid and ask else last

        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "last": last,
            "mid": mid,
            "bid_size": _safe_float(ticker.bidSize),
            "ask_size": _safe_float(ticker.askSize),
            "volume": _safe_float(ticker.volume),
            "high": _safe_float(ticker.high),
            "low": _safe_float(ticker.low),
            "close": _safe_float(ticker.close),
            "time": ticker.time,
        }

    def snapshot_all(self) -> pd.DataFrame:
        """Return snapshots for all subscribed symbols as a DataFrame."""
        rows = [self.snapshot(sym) for sym in self._tickers]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def mid_price(self, symbol: str) -> Optional[float]:
        """Return mid price, or last if bid/ask unavailable."""
        snap = self.snapshot(symbol)
        return snap.get("mid") or snap.get("last")

    # ------------------------------------------------------------------
    # Historical bars (one-shot)
    # ------------------------------------------------------------------

    def historical_bars(
        self,
        symbol: str,
        duration: str = "5 D",
        bar_size: str = "1 day",
        what_to_show: str = "ADJUSTED_LAST",
    ) -> pd.DataFrame:
        """Fetch historical bars (blocking call).

        Parameters
        ----------
        duration : IB duration string, e.g. "5 D", "1 M", "1 Y".
        bar_size : e.g. "1 day", "1 hour", "5 mins".
        """
        ib = self._gw.ib
        contract = self.get_contract(symbol)
        if contract is None:
            return pd.DataFrame()

        bars = ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=True,
            formatDate=1,
        )

        if not bars:
            logger.warning("No historical bars returned for {}", symbol)
            return pd.DataFrame()

        from ib_insync import util
        df = util.df(bars)
        df["symbol"] = symbol
        logger.info("Fetched {} bars for {} ({})", len(df), symbol, bar_size)
        return df

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def subscribed_symbols(self) -> List[str]:
        return list(self._tickers.keys())


def _safe_float(val) -> Optional[float]:
    """Convert IB's nan-like values to None."""
    import math
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
