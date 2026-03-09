"""IB connection manager — connect, disconnect, reconnect, account info.

Wraps ``ib_insync.IB`` with structured logging, automatic reconnection,
and clean shutdown.
"""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Optional

from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import IBConfig


class IBGateway:
    """Manage a single IB TWS / Gateway connection.

    Usage::

        gw = IBGateway(IBConfig())
        gw.connect()
        print(gw.account_summary())
        gw.disconnect()
    """

    def __init__(self, config: IBConfig):
        self.cfg = config
        self._ib: Optional[object] = None  # ib_insync.IB instance
        self._connected = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to TWS / Gateway."""
        from ib_insync import IB

        self._ib = IB()

        # Register event callbacks
        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.errorEvent += self._on_error

        logger.info("Connecting to IB @ {}:{} (clientId={})",
                     self.cfg.host, self.cfg.port, self.cfg.client_id)

        self._ib.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            clientId=self.cfg.client_id,
            timeout=self.cfg.timeout,
            readonly=self.cfg.readonly,
            account=self.cfg.account or "",
        )

        self._connected = True
        acct = self.account_id
        logger.info("Connected to IB — account: {}, server version: {}",
                     acct, self._ib.client.serverVersion())

    def disconnect(self) -> None:
        """Gracefully disconnect."""
        if self._ib and self._connected:
            logger.info("Disconnecting from IB...")
            self._ib.disconnect()
            self._connected = False

    def reconnect(self) -> None:
        """Drop and re-establish connection."""
        self.disconnect()
        self.connect()

    @property
    def is_connected(self) -> bool:
        return self._connected and self._ib is not None and self._ib.isConnected()

    @property
    def ib(self):
        """Raw ``ib_insync.IB`` instance for advanced usage."""
        if not self.is_connected:
            raise ConnectionError("Not connected to IB. Call connect() first.")
        return self._ib

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    @property
    def account_id(self) -> str:
        """First managed account ID."""
        if self._ib and self._ib.managedAccounts():
            return self._ib.managedAccounts()[0]
        return self.cfg.account or "unknown"

    def account_summary(self) -> dict:
        """Fetch key account metrics."""
        ib = self.ib
        summary = {}
        for av in ib.accountSummary(self.account_id):
            summary[av.tag] = av.value
        return summary

    def account_values(self) -> dict:
        """Return current account values as a flat dict."""
        ib = self.ib
        vals = {}
        for av in ib.accountValues(self.account_id):
            key = f"{av.tag}_{av.currency}" if av.currency else av.tag
            vals[key] = av.value
        return vals

    def net_liquidation(self) -> float:
        """Current net liquidation value (USD)."""
        vals = self.account_values()
        nlv = vals.get("NetLiquidation_USD", vals.get("NetLiquidation_BASE", "0"))
        return float(nlv)

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_connected(self):
        logger.info("IB connection established")
        self._connected = True

    def _on_disconnected(self):
        logger.warning("IB connection lost")
        self._connected = False

    def _on_error(self, reqId, errorCode, errorString, contract):
        # Filter informational messages (codes 2000-2999)
        if 2000 <= errorCode < 3000:
            logger.debug("IB info [{}]: {}", errorCode, errorString)
        else:
            logger.error("IB error [{}] reqId={}: {} | contract={}",
                         errorCode, reqId, errorString, contract)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def sleep(self, seconds: float = 0):
        """Wrapper around ``ib.sleep()`` to process IB messages."""
        if self._ib:
            self._ib.sleep(seconds)
