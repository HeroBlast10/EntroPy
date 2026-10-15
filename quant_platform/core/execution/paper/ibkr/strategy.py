"""Signal-to-order strategy runner.

Implements a simple loop:
1. Fetch latest prices from IB
2. Compute target weights (from pre-computed factor signals or live)
3. Diff against current positions → generate rebalance orders
4. Submit orders through the risk-gated OrderManager
5. Log everything, sleep, repeat

This is a **demo** strategy — production would add order management,
partial fill handling, slippage monitoring, etc.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import PaperTradingConfig
from quant_platform.core.execution.paper.ibkr.execution import OrderManager
from quant_platform.core.execution.paper.ibkr.gateway import IBGateway
from quant_platform.core.execution.paper.ibkr.market_data import MarketDataManager
from quant_platform.core.execution.paper.ibkr.portfolio import PortfolioTracker
from quant_platform.core.execution.paper.ibkr.risk import RiskManager


class PaperTradingStrategy:
    """Main strategy loop for the IB paper-trading demo.

    Orchestrates: gateway → market data → signal → rebalance → orders → log.

    Usage::

        strategy = PaperTradingStrategy(config)
        strategy.start()      # blocking loop
        # or
        strategy.run_once()   # single rebalance cycle
    """

    def __init__(self, config: PaperTradingConfig):
        config.validate()
        self.cfg = config
        self.gateway = IBGateway(config.ib)
        self.risk_mgr = RiskManager(config.risk)
        self.md: Optional[MarketDataManager] = None
        self.orders: Optional[OrderManager] = None
        self.portfolio: Optional[PortfolioTracker] = None

        self._running = False
        self._cycle_count = 0
        self._target_weights: Dict[str, float] = {}

        # Logging
        self._log_dir = Path(config.log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._state_dir = Path(config.state_dir)
        self._state_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to IB and enter the main rebalance loop."""
        logger.info("=" * 60)
        logger.info("EntroPy Paper Trading — starting")
        logger.info("Tickers: {}", self.cfg.strategy.tickers)
        logger.info("Rebalance interval: {}s", self.cfg.strategy.rebalance_interval_sec)
        logger.info("Dry run: {}", self.cfg.strategy.dry_run)
        logger.info("=" * 60)

        try:
            # Connect
            self.gateway.connect()
            self.md = MarketDataManager(self.gateway)
            self.orders = OrderManager(self.gateway, self.risk_mgr, self.cfg)
            self.portfolio = PortfolioTracker(self.gateway)
            self.portfolio.subscribe_pnl()

            # Subscribe to market data
            self.md.subscribe(self.cfg.strategy.tickers)

            # Wait for initial data
            logger.info("Waiting 5s for market data to populate...")
            self.gateway.sleep(5)

            # Log account info
            logger.info("Account base currency: {}", self.gateway.base_currency())

            # Main loop
            self._running = True
            while self._running:
                try:
                    self.run_once()
                except KeyboardInterrupt:
                    logger.info("Keyboard interrupt — stopping")
                    break
                except Exception as exc:
                    logger.error("Strategy cycle error: {}", exc)
                    # Don't crash — log and continue
                    if self.risk_mgr.is_killed:
                        logger.critical("Kill switch active — stopping loop")
                        break

                # Sleep between cycles (IB event processing continues)
                interval = self.cfg.strategy.rebalance_interval_sec
                logger.info("Next rebalance in {}s...", interval)
                self.gateway.sleep(interval)

        except Exception as exc:
            logger.critical("Fatal error: {}", exc)
            raise
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean up: cancel orders, unsubscribe, disconnect, save logs."""
        logger.info("Shutting down...")

        try:
            if self.orders:
                try:
                    if self.gateway.is_connected:
                        self.orders.cancel_all()
                finally:
                    self.orders.save_fill_log()
        finally:
            try:
                if self.gateway.is_connected:
                    try:
                        if self.md:
                            self.md.unsubscribe_all()
                    finally:
                        if self.portfolio:
                            self.portfolio.close()
            finally:
                self.gateway.disconnect()
                self._save_state()
                self._running = False
        logger.info("Shutdown complete")

    def stop(self) -> None:
        """Signal the main loop to exit."""
        self._running = False

    # ------------------------------------------------------------------
    # Single rebalance cycle
    # ------------------------------------------------------------------

    def run_once(self) -> None:
        """Execute one rebalance cycle: signal → target → diff → orders."""
        self._cycle_count += 1
        cycle_ts = dt.datetime.now()
        logger.info("--- Cycle #{} @ {} ---", self._cycle_count,
                     cycle_ts.strftime("%Y-%m-%d %H:%M:%S"))

        # 1) Refresh market data
        self.gateway.sleep(0)
        snapshots = self.md.snapshot_all()

        if snapshots.empty:
            logger.warning("No market data — skipping cycle")
            return

        logger.info("Market data:\n{}", snapshots[
            ["symbol", "bid", "ask", "last", "volume"]
        ].to_string(index=False))

        # 2) Compute target weights
        prices = {}
        for _, row in snapshots.iterrows():
            sym = row["symbol"]
            mid = row.get("mid")
            if mid is not None and math.isfinite(mid) and mid > 0:
                prices[sym] = mid

        self._target_weights = self._compute_signal(prices)
        if not self._target_weights:
            logger.info("No explicit valid weights — skipping rebalance")
            return
        logger.info("Target weights: {}", {k: f"{v:.2%}" for k, v in
                     self._target_weights.items()})

        # 3) Compute rebalance trades
        capital = self.cfg.strategy.target_capital
        pnl = self.portfolio.pnl_summary()
        # Account PnL is in base currency; USD risk limits cannot use HKD values.
        self.risk_mgr.update_daily_pnl(pnl["daily_pnl"] if pnl["currency"] == "USD" else None)
        submitting = self.cfg.strategy.enable_paper_orders
        if submitting:
            if pnl["currency"] != "USD":
                raise ValueError("Paper order mode requires USD base currency for USD loss limits.")
            capital = min(capital, self.gateway.net_liquidation("USD"))
            now = dt.datetime.now(dt.timezone.utc)
            for _, row in snapshots.iterrows():
                if row["symbol"] not in self._target_weights:
                    continue
                quote_time = row.get("time")
                age = (now - quote_time).total_seconds() if quote_time is not None and not pd.isna(quote_time) and quote_time.tzinfo else float("inf")
                if row["market_data_type"] != 1 or not 0 <= age <= 120:
                    raise ValueError("Paper orders require fresh real-time quotes; delayed/frozen quotes are diagnostic only.")
        rebal_orders = self.portfolio.compute_rebalance(
            self._target_weights, prices, capital,
        )

        if not rebal_orders:
            logger.info("No rebalance trades needed")
            self._log_cycle(cycle_ts, snapshots, rebal_orders)
            return

        # 4) Submit orders
        for order_spec in rebal_orders:
            sym = order_spec["symbol"]
            side = order_spec["side"]
            qty = order_spec["shares"]
            ref_price = order_spec["price"]

            order_type = self.cfg.strategy.order_type.upper()

            if order_type == "LMT":
                offset = self.cfg.strategy.limit_offset_bps / 10_000
                if side == "BUY":
                    lmt = ref_price * (1 + offset)
                else:
                    lmt = ref_price * (1 - offset)
                self.orders.place_limit_order(sym, side, qty, round(lmt, 2))
            elif order_type == "ADAPTIVE":
                self.orders.place_adaptive_order(sym, side, qty, ref_price)
            else:
                self.orders.place_market_order(sym, side, qty, ref_price)

        # 5) Wait for fills
        logger.info("Waiting 3s for fills...")
        self.gateway.sleep(3)

        # 6) Log portfolio status
        logger.info("\n{}", self.portfolio.status_report())
        logger.info("Risk status: {}", self.risk_mgr.status())

        # 7) Refresh the broker's daily PnL
        pnl = self.portfolio.pnl_summary()
        self.risk_mgr.update_daily_pnl(pnl["daily_pnl"] if pnl["currency"] == "USD" else None)

        self._log_cycle(cycle_ts, snapshots, rebal_orders)

    # ------------------------------------------------------------------
    # Signal computation
    # ------------------------------------------------------------------

    def _compute_signal(self, prices: Dict[str, float]) -> Dict[str, float]:
        """Load explicit reviewed weights; missing signals never trigger buying."""
        valid = {k: v for k, v in prices.items() if v is not None and math.isfinite(v) and v > 0}
        if not valid:
            return {}

        # Try to load pre-computed factor scores
        target = self._load_factor_weights(valid)
        if target:
            return target

        return {}

    def _load_factor_weights(self, prices: Dict[str, float]) -> Optional[Dict[str, float]]:
        """Validate the explicit dated weights file without renormalising cash."""
        try:
            from quant_platform.core.utils.io import load_parquet

            if not self.cfg.strategy.weights_file:
                return None
            path = Path(self.cfg.strategy.weights_file)
            df = load_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            latest_date = df["date"].max()
            from quant_platform.core.execution.paper.ibkr.risk import trading_date
            age = (trading_date() - latest_date.date()).days
            if not 0 <= age <= self.cfg.strategy.max_weights_age_days:
                raise ValueError("Weights are future-dated or stale.")
            latest = df[df["date"] == latest_date]
            if latest["ticker"].duplicated().any():
                raise ValueError("Duplicate tickers in weights file.")

            weights = {}
            for _, row in latest.iterrows():
                tkr = row["ticker"]
                w = float(row["weight"])
                if not math.isfinite(w) or w < 0:
                    raise ValueError("Weights must be finite and nonnegative.")
                if w > 0 and tkr not in prices:
                    raise ValueError("A target ticker has no valid subscribed quote.")
                if tkr in prices:
                    weights[tkr] = w

            if weights:
                total = sum(weights.values())
                if total > 1.0 + 1e-8:
                    raise ValueError("Weights exceed 100%; leverage is disabled.")
                logger.info("Loaded factor weights from {} ({})",
                            path.name, latest_date.date())
                return weights
        except Exception as exc:
            logger.debug("Factor weight load failed: {}", exc)

        return None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_cycle(self, ts, snapshots, orders):
        """Append cycle summary to daily log file."""
        log_file = self._log_dir / f"cycles_{ts.strftime('%Y%m%d')}.jsonl"
        record = {
            "cycle": self._cycle_count,
            "timestamp": ts.isoformat(),
            "n_symbols": len(snapshots),
            "n_orders": len(orders),
            "orders": orders,
            "risk": self.risk_mgr.status(),
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _save_state(self):
        """Persist final state for post-session analysis."""
        state_file = self._state_dir / f"state_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        state = {
            "cycles": self._cycle_count,
            "target_weights": self._target_weights,
            "risk_status": self.risk_mgr.status(),
            "session": {
                "trades": self.risk_mgr.session.trades_count,
                "notional": self.risk_mgr.session.traded_notional,
                "pnl": self.risk_mgr.session.total_pnl,
            },
        }
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        logger.info("State saved → {}", state_file)
