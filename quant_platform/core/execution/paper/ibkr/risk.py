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
import math
from zoneinfo import ZoneInfo
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import RiskLimits


def trading_date() -> dt.date:
    return dt.datetime.now(ZoneInfo("America/New_York")).date()


@dataclass
class SessionState:
    """Mutable counters for the current trading session."""

    date: dt.date = field(default_factory=trading_date)
    trades_count: int = 0
    traded_notional: float = 0.0
    realised_pnl: float = 0.0
    unrealised_pnl: float = 0.0
    daily_pnl: Optional[float] = None

    @property
    def total_pnl(self) -> Optional[float]:
        return self.daily_pnl

    def reset_if_new_day(self):
        today = trading_date()
        if today != self.date:
            logger.info("New session day {} — resetting counters", today)
            self.date = today
            self.trades_count = 0
            self.traded_notional = 0.0
            self.realised_pnl = 0.0
            self.unrealised_pnl = 0.0
            self.daily_pnl = None


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

    def _check_daily_loss(self) -> Tuple[bool, str]:
        if not self.limits.check_session_limits:
            return True, ""
        if self.session.total_pnl is None:
            return False, "Daily PnL unavailable; increasing risk is blocked"
        if self.session.total_pnl <= -self.limits.max_daily_loss:
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
        current_position_shares: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Run all pre-trade risk checks.

        Returns
        -------
        (True, "") if all checks pass, (False, reason) otherwise.
        """
        self.session.reset_if_new_day()

        if side not in {"BUY", "SELL"} or not isinstance(shares, int) or shares <= 0:
            return False, "Invalid side or share quantity"
        if not math.isfinite(price) or price <= 0 or not math.isfinite(existing_position_notional):
            return False, "Invalid reference price or position exposure"
        notional = shares * price
        projected = existing_position_notional + (notional if side == "BUY" else -notional)
        reducing = (existing_position_notional != 0 and
                    existing_position_notional * projected >= 0 and
                    abs(projected) < abs(existing_position_notional))
        if self.limits.long_only and projected < -1e-6:
            return False, "Long-only setup cannot create a short position"

        position_check = (True, "")
        if self.limits.check_position_limits and not reducing:
            if abs(projected) > self.limits.max_position_notional:
                position_check = (False, "Projected position exceeds notional limit")
            if current_position_shares is not None:
                projected_shares = current_position_shares + (shares if side == "BUY" else -shares)
                if abs(projected_shares) > self.limits.max_position_shares:
                    position_check = (False, "Projected position exceeds share limit")

        checks = [
            self._check_kill_switch(),
            self._check_order_notional(notional),
            self._check_order_shares(abs(shares)),
            self._check_position_count(current_positions) if existing_position_notional == 0 else (True, ""),
            position_check,
            self._check_daily_loss() if not reducing else (True, ""),
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
                self.session.total_pnl is not None and
                self.session.total_pnl <= -self.limits.max_daily_loss):
            self.activate_kill_switch(
                f"Daily loss limit breached: ${abs(self.session.total_pnl):,.0f}"
            )

    def update_unrealised_pnl(self, unrealised: float):
        """Update unrealised PnL from portfolio snapshot."""
        self.session.unrealised_pnl = unrealised

    def update_daily_pnl(self, daily: Optional[float]) -> None:
        """Use broker daily PnL, not lifetime unrealised PnL or a synthetic zero."""
        self.session.reset_if_new_day()
        self.session.daily_pnl = daily if daily is not None and math.isfinite(daily) and abs(daily) < 1e100 else None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        """Return current risk state as a dict (for logging / UI)."""
        return {
            "kill_switch": self.limits.kill_switch,
            "daily_trades": self.session.trades_count,
            "daily_notional": f"${self.session.traded_notional:,.0f}",
            "daily_pnl": (f"${self.session.total_pnl:,.0f}" if self.session.total_pnl is not None else None),
            "limits": {
                "max_order_notional": self.limits.max_order_notional,
                "max_daily_loss": self.limits.max_daily_loss,
                "max_positions": self.limits.max_positions,
            },
        }
