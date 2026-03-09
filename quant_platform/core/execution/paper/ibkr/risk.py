"""Pre-trade risk checks and session-level risk controls.

Every check returns a ``(passed: bool, reason: str)`` tuple.
The ``RiskManager`` aggregates checks and provides a kill-switch.

Design:
- Each check is a standalone function (unit-testable).
- ``RiskManager`` holds mutable session state (daily PnL, trade count, etc.)
  and is the single gate through which all orders must pass.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import RiskLimits


@dataclass
class SessionState:
    """Mutable counters for the current trading session."""

    date: dt.date = field(default_factory=dt.date.today)
    trades_count: int = 0
    traded_notional: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0

    @property
    def total_pnl(self) -> float:
        return self.realised_pnl + self.unrealised_pnl

    def reset_if_new_day(self):
        today = dt.date.today()
        if today != self.date:
            logger.info("New session day {} — resetting counters", today)
            self.date = today
            self.trades_count = 0
            self.traded_notional = 0.0
            self.realised_pnl = 0.0
            self.unrealised_pnl = 0.0


class RiskManager:
    """Centralised pre-trade risk gate.

    Usage::

        rm = RiskManager(limits)
        ok, reason = rm.check_order(side="BUY", shares=100, price=150.0, ...)
        if not ok:
            logger.warning("Order rejected: {}", reason)
    """

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.session = SessionState()
        self._kill_reason: Optional[str] = None

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def activate_kill_switch(self, reason: str = "Manual kill switch"):
        """Stop all new orders immediately."""
        self.limits.kill_switch = True
        self._kill_reason = reason
        logger.critical("KILL SWITCH ACTIVATED: {}", reason)

    def deactivate_kill_switch(self):
        """Re-enable order flow."""
        self.limits.kill_switch = False
        self._kill_reason = None
        logger.warning("Kill switch deactivated — order flow resumed")

    @property
    def is_killed(self) -> bool:
        return self.limits.kill_switch

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_kill_switch(self) -> Tuple[bool, str]:
        if self.limits.kill_switch:
            return False, f"Kill switch active: {self._kill_reason or 'no reason'}"
        return True, ""

    def _check_order_notional(self, notional: float) -> Tuple[bool, str]:
        if not self.limits.check_order_limits:
            return True, ""
        if notional > self.limits.max_order_notional:
            return False, (f"Order notional ${notional:,.0f} exceeds limit "
                           f"${self.limits.max_order_notional:,.0f}")
        return True, ""

    def _check_order_shares(self, shares: int) -> Tuple[bool, str]:
        if not self.limits.check_order_limits:
            return True, ""
        if shares > self.limits.max_order_shares:
            return False, (f"Order shares {shares:,} exceeds limit "
                           f"{self.limits.max_order_shares:,}")
        return True, ""

    def _check_position_count(self, current_positions: int) -> Tuple[bool, str]:
        if not self.limits.check_position_limits:
            return True, ""
        if current_positions >= self.limits.max_positions:
            return False, (f"Position count {current_positions} >= limit "
                           f"{self.limits.max_positions}")
        return True, ""

    def _check_position_notional(
        self, existing_notional: float, order_notional: float
    ) -> Tuple[bool, str]:
        if not self.limits.check_position_limits:
            return True, ""
        projected = existing_notional + order_notional
        if projected > self.limits.max_position_notional:
            return False, (f"Projected position ${projected:,.0f} exceeds limit "
                           f"${self.limits.max_position_notional:,.0f}")
        return True, ""

    def _check_daily_loss(self) -> Tuple[bool, str]:
        if not self.limits.check_session_limits:
            return True, ""
        if self.session.total_pnl < -self.limits.max_daily_loss:
            return False, (f"Daily loss ${abs(self.session.total_pnl):,.0f} exceeds limit "
                           f"${self.limits.max_daily_loss:,.0f}")
        return True, ""

    def _check_daily_trades(self) -> Tuple[bool, str]:
        if not self.limits.check_session_limits:
            return True, ""
        if self.session.trades_count >= self.limits.max_daily_trades:
            return False, (f"Daily trades {self.session.trades_count} >= limit "
                           f"{self.limits.max_daily_trades}")
        return True, ""

    def _check_daily_notional(self, order_notional: float) -> Tuple[bool, str]:
        if not self.limits.check_session_limits:
            return True, ""
        projected = self.session.traded_notional + order_notional
        if projected > self.limits.max_daily_notional:
            return False, (f"Daily traded notional ${projected:,.0f} exceeds limit "
                           f"${self.limits.max_daily_notional:,.0f}")
        return True, ""

    # ------------------------------------------------------------------
    # Aggregated check
    # ------------------------------------------------------------------

    def check_order(
        self,
        side: str,
        shares: int,
        price: float,
        current_positions: int = 0,
        existing_position_notional: float = 0.0,
    ) -> Tuple[bool, str]:
        """Run all pre-trade risk checks.

        Returns
        -------
        (True, "") if all checks pass, (False, reason) otherwise.
        """
        self.session.reset_if_new_day()

        notional = abs(shares * price)

        checks = [
            self._check_kill_switch(),
            self._check_order_notional(notional),
            self._check_order_shares(abs(shares)),
            self._check_position_count(current_positions),
            self._check_position_notional(existing_position_notional, notional),
            self._check_daily_loss(),
            self._check_daily_trades(),
            self._check_daily_notional(notional),
        ]

        for passed, reason in checks:
            if not passed:
                logger.warning("RISK REJECT | {} {} @ ${:.2f} | {}",
                               side, shares, price, reason)
                return False, reason

        return True, ""

    # ------------------------------------------------------------------
    # Post-trade updates
    # ------------------------------------------------------------------

    def record_fill(self, notional: float, pnl: float = 0.0):
        """Update session counters after a fill."""
        self.session.trades_count += 1
        self.session.traded_notional += abs(notional)
        self.session.realised_pnl += pnl

        # Auto kill-switch on daily loss breach
        if (self.limits.check_session_limits and
                self.session.total_pnl < -self.limits.max_daily_loss):
            self.activate_kill_switch(
                f"Daily loss limit breached: ${abs(self.session.total_pnl):,.0f}"
            )

    def update_unrealised_pnl(self, unrealised: float):
        """Update unrealised PnL from portfolio snapshot."""
        self.session.unrealised_pnl = unrealised

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return current risk state as a dict (for logging / UI)."""
        return {
            "kill_switch": self.limits.kill_switch,
            "daily_trades": self.session.trades_count,
            "daily_notional": f"${self.session.traded_notional:,.0f}",
            "daily_pnl": f"${self.session.total_pnl:,.0f}",
            "limits": {
                "max_order_notional": self.limits.max_order_notional,
                "max_daily_loss": self.limits.max_daily_loss,
                "max_positions": self.limits.max_positions,
            },
        }
