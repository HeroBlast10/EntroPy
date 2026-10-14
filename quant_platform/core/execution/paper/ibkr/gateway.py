"""IB connection manager — connect, disconnect, reconnect, account info.

Wraps ``ib_insync.IB`` with structured logging, explicit reconnect,
paper-account validation and clean shutdown.
"""

from __future__ import annotations

import math
import re
from typing import Optional

from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import IBConfig


def readable_message(message: str) -> str:
    """IB may return literal Unicode escapes in Chinese messages."""
    return re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), message)


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
        self.messages: list[dict] = []

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to TWS / Gateway."""
        self.cfg.validate()
        from ib_insync import IB

        self._ib = IB()

        # Register event callbacks
        self._ib.connectedEvent += self._on_connected
        self._ib.disconnectedEvent += self._on_disconnected
        self._ib.errorEvent += self._on_error

        logger.info("Connecting to IB @ {}:{} (clientId={})",
                     self.cfg.host, self.cfg.port, self.cfg.client_id)

        self._ib.RequestTimeout = self.cfg.timeout
        try:
            self._ib.connect(
                host=self.cfg.host, port=self.cfg.port,
                clientId=self.cfg.client_id, timeout=self.cfg.timeout,
                readonly=self.cfg.readonly, account=self.cfg.account or "",
            )
            self._connected = True
            self.assert_paper_account()
            logger.info("Connected to paper account; server version: {}",
                        self._ib.client.serverVersion())
        except BaseException:
            self._ib.disconnect()
            self._connected = False
            raise

    def disconnect(self) -> None:
        """Gracefully disconnect."""
        if self._ib:
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
        """Resolve the selected account; never silently choose among accounts."""
        accounts = self._ib.managedAccounts() if self._ib else []
        if self.cfg.account:
            if self._ib is not None and self.cfg.account not in accounts:
                raise ValueError("Selected account is not available in this session.")
            return self.cfg.account
        if len(accounts) != 1:
            raise ValueError("Select an explicit DU account when zero or multiple accounts are available.")
        return accounts[0]

    def assert_paper_account(self) -> str:
        account = self.account_id
        if not account.startswith("DU"):
            raise ValueError("Live/non-DU accounts are blocked by this paper adapter.")
        return account

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

    def base_currency(self) -> str:
        """Actual account base currency; never treat BASE as USD implicitly."""
        for av in self.ib.accountValues(self.account_id):
            if av.tag == "Currency" and av.currency == "BASE" and len(av.value) == 3 and av.value.isalpha():
                return av.value
        for av in self.ib.accountSummary(self.account_id):
            if av.tag == "NetLiquidation" and av.currency not in {"", "BASE"}:
                return av.currency
        return "unknown"

    def net_liquidation(self, currency: str = "USD") -> float:
        """Net liquidation in the requested currency, or fail if unavailable."""
        vals = self.account_values()
        nlv = vals.get(f"NetLiquidation_{currency}")
        if nlv is None and self.base_currency() == currency:
            nlv = vals.get("NetLiquidation_BASE")
        if nlv is None or not math.isfinite(float(nlv)):
            raise ValueError(f"NetLiquidation in {currency} is unavailable; no implicit FX conversion.")
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
        errorString = readable_message(errorString)
        self.messages.append({"code": errorCode, "message": errorString})
        if errorCode == 2172:
            logger.warning("IB version notice [{}]: {}", errorCode, errorString)
            return
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
