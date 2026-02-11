"""Paper trading configuration and risk-limit definitions.

All tuneable knobs for the IB paper-trading demo live here so they can
be changed in one place (or loaded from YAML in future).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class IBConfig:
    """IB TWS / Gateway connection parameters."""

    host: str = "127.0.0.1"
    port: int = 7497          # 7497 = TWS paper, 4002 = Gateway paper
    client_id: int = 1
    timeout: float = 30.0     # seconds to wait for connection
    readonly: bool = False     # True = market-data only, no orders
    account: str = ""          # leave blank to auto-detect first account


@dataclass
class RiskLimits:
    """Pre-trade and session-level risk controls.

    Every limit has an ``enabled`` flag so individual checks can be
    toggled off during development without removing the code.
    """

    # --- Kill switch ---
    kill_switch: bool = False          # if True, reject ALL new orders

    # --- Per-order limits ---
    max_order_notional: float = 50_000.0    # USD
    max_order_shares: int = 5_000
    max_order_pct_adv: float = 0.01         # 1 % of ADV

    # --- Position limits ---
    max_position_notional: float = 100_000.0
    max_position_shares: int = 10_000
    max_positions: int = 20                  # total number of open positions

    # --- Session limits ---
    max_daily_loss: float = 10_000.0         # USD — hard stop for the day
    max_daily_trades: int = 200
    max_daily_notional: float = 1_000_000.0  # cumulative traded notional

    # --- Enable/disable groups ---
    check_order_limits: bool = True
    check_position_limits: bool = True
    check_session_limits: bool = True


@dataclass
class StrategyConfig:
    """Configuration for the demo signal-to-order strategy."""

    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "GOOGL", "AMZN", "JPM",
    ])
    signal_col: str = "MOM_12_1M"
    rebalance_interval_sec: int = 300   # 5 minutes
    target_capital: float = 100_000.0   # paper account equity to assume
    order_type: str = "MKT"             # MKT | LMT | ADAPTIVE
    limit_offset_bps: float = 5.0       # for LMT orders, offset from mid
    dry_run: bool = False               # if True, log orders but don't submit


@dataclass
class PaperTradingConfig:
    """Top-level config aggregating all sub-configs."""

    ib: IBConfig = field(default_factory=IBConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    log_dir: str = "data/live/logs"
    state_dir: str = "data/live/state"
