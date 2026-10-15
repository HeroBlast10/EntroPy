"""Paper trading configuration and risk-limit definitions.

Load the paper-only research profile with ``load_paper_config``. No credentials
are stored here; only an explicitly enabled paper session may submit orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
import math
from typing import List


@dataclass
class IBConfig:
    """IB TWS / Gateway connection parameters."""

    host: str = "127.0.0.1"
    port: int = 4002          # 7497 = TWS paper, 4002 = Gateway paper
    client_id: int = 17
    timeout: float = 30.0     # seconds to wait for connection
    readonly: bool = True
    account: str = ""          # auto-select only when exactly one paper account exists
    market_data_type: int = 3  # delayed; entitlement must still be verified

    def validate(self) -> None:
        if self.host not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("This paper setup only supports a local Gateway/TWS.")
        if self.port not in {4002, 7497}:
            raise ValueError("Only paper ports 4002 and 7497 are allowed.")
        if self.client_id <= 0:
            raise ValueError("Use a positive dedicated client ID, not client ID 0.")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be finite and positive.")
        if self.market_data_type not in {1, 2, 3, 4}:
            raise ValueError("market_data_type must be 1, 2, 3 or 4.")
        if self.account and not self.account.startswith("DU"):
            raise ValueError("An explicitly selected account must be a DU paper account.")


@dataclass
class RiskLimits:
    """Pre-trade and session-level risk controls.

    Every limit has an ``enabled`` flag so individual checks can be
    toggled off during development without removing the code.
    """

    # --- Kill switch ---
    kill_switch: bool = False          # if True, reject ALL new orders

    # --- Per-order limits ---
    max_order_notional: float = 1_200.0    # provisional USD research limits
    max_order_shares: int = 500
    max_order_pct_adv: float = 0.01         # 1 % of ADV

    # --- Position limits ---
    max_position_notional: float = 1_200.0
    max_position_shares: int = 500
    max_positions: int = 10
    long_only: bool = True

    # --- Session limits ---
    max_daily_loss: float = 120.0           # USD, provisional paper limit
    max_daily_trades: int = 20
    max_daily_notional: float = 3_600.0

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
    rebalance_interval_sec: int = 86_400  # demo loop only, not an exchange scheduler
    target_capital: float = 12_000.0     # provisional USD cap, NOT a live HKD conversion
    order_type: str = "LMT"             # MKT | LMT | ADAPTIVE
    limit_offset_bps: float = 5.0       # for LMT orders, offset from mid
    dry_run: bool = True
    enable_paper_orders: bool = False   # separate explicit opt-in
    weights_file: str = ""              # no automatic file selection / equal-weight buying
    max_weights_age_days: int = 7


@dataclass
class PaperTradingConfig:
    """Top-level config aggregating all sub-configs."""

    ib: IBConfig = field(default_factory=IBConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    log_dir: str = "data/live/logs"
    state_dir: str = "data/live/state"

    def validate(self) -> None:
        self.ib.validate()
        if not math.isfinite(self.strategy.target_capital) or self.strategy.target_capital <= 0:
            raise ValueError("target_capital must be finite and positive USD.")
        if self.strategy.rebalance_interval_sec <= 0 or self.strategy.max_weights_age_days < 0:
            raise ValueError("Invalid strategy interval or weights age.")
        if not self.strategy.tickers or any(not t.strip() for t in self.strategy.tickers):
            raise ValueError("At least one nonempty probe/strategy ticker is required.")
        if self.strategy.order_type not in {"MKT", "LMT", "ADAPTIVE"}:
            raise ValueError("Invalid order type.")
        if not math.isfinite(self.strategy.limit_offset_bps) or self.strategy.limit_offset_bps < 0:
            raise ValueError("Invalid limit price offset.")
        for f in fields(self.risk):
            value = getattr(self.risk, f.name)
            if not isinstance(value, bool) and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"risk.{f.name} must be finite and positive.")
        if self.strategy.enable_paper_orders:
            if self.ib.readonly or self.strategy.dry_run or not self.ib.account:
                raise ValueError("Paper orders require an explicit DU account, readwrite and no-dry-run.")
            if not self.strategy.weights_file:
                raise ValueError("Paper orders require an explicit, reviewed weights_file.")


def load_paper_config(path: str | Path) -> PaperTradingConfig:
    """Load a strict YAML config; reject unknown keys and mistyped flags."""
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config must be a YAML mapping.")
    allowed = {f.name for f in fields(PaperTradingConfig)}
    if set(raw) - allowed:
        raise ValueError(f"Unknown config keys: {sorted(set(raw) - allowed)}")
    kwargs = {}
    for name, cls in [("ib", IBConfig), ("risk", RiskLimits), ("strategy", StrategyConfig)]:
        section = raw.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a mapping.")
        defaults = cls()
        known = {f.name for f in fields(cls)}
        if set(section) - known:
            raise ValueError(f"Unknown {name} keys: {sorted(set(section) - known)}")
        for key, value in section.items():
            default = getattr(defaults, key)
            expected = type(default)
            valid = (type(value) in {int, float} if expected is float else type(value) is expected)
            if not valid or (expected is list and not all(isinstance(x, str) for x in value)):
                raise ValueError(f"Invalid type for {name}.{key}.")
        kwargs[name] = cls(**section)
    for name in ("log_dir", "state_dir"):
        if name in raw:
            if not isinstance(raw[name], str):
                raise ValueError(f"{name} must be a string.")
            kwargs[name] = raw[name]
    config = PaperTradingConfig(**kwargs)
    config.validate()
    return config
