"""Order placement, status tracking, and fill callbacks.

Provides ``OrderManager`` which is the single interface for submitting,
modifying, and cancelling orders through IB.  Every order goes through
the ``RiskManager`` gate before submission.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from quant_platform.core.execution.paper.ibkr.config import PaperTradingConfig
from quant_platform.core.execution.paper.ibkr.risk import RiskManager


class OrderManager:
    """Submit and track orders via IB.

    Usage::

        om = OrderManager(gateway, risk_manager, config)
        trade = om.place_market_order("AAPL", "BUY", 100)
        om.place_limit_order("MSFT", "SELL", 50, limit_price=410.0)
        print(om.open_orders())
        om.cancel_all()
    """

    def __init__(self, gateway, risk_manager: RiskManager, config: PaperTradingConfig):
        self._gw = gateway
        self._risk = risk_manager
        self._cfg = config
        self._trades: List[object] = []       # ib_insync.Trade objects
        self._fill_log: List[Dict] = []       # structured fill records
        self._order_id_map: Dict[int, str] = {}  # orderId → symbol

        # Wire up fill callback
        ib = self._gw.ib
        ib.orderStatusEvent += self._on_order_status
        ib.execDetailsEvent += self._on_exec_details

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(
        self, symbol: str, side: str, quantity: int,
        ref_price: Optional[float] = None,
    ) -> Optional[object]:
        """Place a market order after risk checks.

        Parameters
        ----------
        symbol : ticker
        side : "BUY" or "SELL"
        quantity : number of shares (positive)
        ref_price : reference price for risk check (mid / last)

        Returns the ib_insync Trade object, or None if rejected.
        """
        from ib_insync import MarketOrder
        order = MarketOrder(side, quantity)
        return self._submit(symbol, order, ref_price)

    def place_limit_order(
        self, symbol: str, side: str, quantity: int,
        limit_price: float,
    ) -> Optional[object]:
        """Place a limit order after risk checks."""
        from ib_insync import LimitOrder
        order = LimitOrder(side, quantity, limit_price)
        return self._submit(symbol, order, limit_price)

    def place_adaptive_order(
        self, symbol: str, side: str, quantity: int,
        ref_price: Optional[float] = None,
    ) -> Optional[object]:
        """Place an IB Adaptive algo order (patient fill)."""
        from ib_insync import Order
        order = Order()
        order.action = side
        order.totalQuantity = quantity
        order.orderType = "MKT"
        order.algoStrategy = "Adaptive"
        order.algoParams = [
            {"tag": "adaptivePriority", "value": "Patient"},
        ]
        return self._submit(symbol, order, ref_price)

    # ------------------------------------------------------------------
    # Internal submit
    # ------------------------------------------------------------------

    def _submit(
        self, symbol: str, order, ref_price: Optional[float]
    ) -> Optional[object]:
        """Shared logic: risk check → submit → log."""
        from quant_platform.core.execution.paper.ibkr.market_data import MarketDataManager

        ib = self._gw.ib

        # Get contract
        from ib_insync import Stock
        contract = Stock(symbol, "SMART", "USD")
        qualified = ib.qualifyContracts(contract)
        if not qualified:
            logger.error("Cannot qualify contract for {}", symbol)
            return None
        contract = qualified[0]

        # Estimate price for risk check
        price = ref_price or 100.0  # fallback; caller should provide mid

        # --- Risk gate ---
        positions = self._count_positions()
        existing_notional = self._position_notional(symbol)

        passed, reason = self._risk.check_order(
            side=order.action,
            shares=int(order.totalQuantity),
            price=price,
            current_positions=positions,
            existing_position_notional=existing_notional,
        )
        if not passed:
            logger.warning("ORDER REJECTED | {} {} {} @ ~${:.2f} | {}",
                           order.action, order.totalQuantity, symbol, price, reason)
            return None

        # --- Dry run ---
        if self._cfg.strategy.dry_run:
            logger.info("DRY RUN | {} {} {} @ ~${:.2f} | order_type={}",
                        order.action, order.totalQuantity, symbol, price,
                        order.orderType)
            return None

        # --- Submit ---
        trade = ib.placeOrder(contract, order)
        self._trades.append(trade)
        self._order_id_map[trade.order.orderId] = symbol

        logger.info("ORDER SUBMITTED | {} {} {} @ {} | orderId={}",
                     order.action, order.totalQuantity, symbol,
                     order.orderType, trade.order.orderId)
        return trade

    # ------------------------------------------------------------------
    # Cancel / modify
    # ------------------------------------------------------------------

    def cancel_order(self, trade) -> None:
        """Cancel a specific order."""
        ib = self._gw.ib
        ib.cancelOrder(trade.order)
        logger.info("ORDER CANCELLED | orderId={}", trade.order.orderId)

    def cancel_all(self) -> None:
        """Cancel all open orders."""
        ib = self._gw.ib
        ib.reqGlobalCancel()
        logger.warning("ALL ORDERS CANCELLED (global cancel)")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def open_orders(self) -> List[Dict]:
        """Return list of currently open orders."""
        ib = self._gw.ib
        result = []
        for trade in ib.openTrades():
            result.append({
                "orderId": trade.order.orderId,
                "symbol": trade.contract.symbol,
                "action": trade.order.action,
                "quantity": trade.order.totalQuantity,
                "order_type": trade.order.orderType,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
            })
        return result

    def fill_log_df(self) -> pd.DataFrame:
        """Return all fills as a DataFrame."""
        if not self._fill_log:
            return pd.DataFrame()
        return pd.DataFrame(self._fill_log)

    def save_fill_log(self, path: Optional[Path] = None) -> Path:
        """Persist fill log to JSON."""
        if path is None:
            log_dir = Path(self._cfg.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = log_dir / f"fills_{ts}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._fill_log, indent=2, default=str),
                        encoding="utf-8")
        logger.info("Fill log saved → {} ({} fills)", path, len(self._fill_log))
        return path

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_order_status(self, trade):
        status = trade.orderStatus
        sym = trade.contract.symbol if trade.contract else "?"
        logger.debug("ORDER STATUS | {} {} | status={} filled={} remaining={}",
                      sym, trade.order.orderId, status.status,
                      status.filled, status.remaining)

    def _on_exec_details(self, trade, fill):
        """Called on each execution (partial or full fill)."""
        sym = trade.contract.symbol if trade.contract else "?"
        exec_info = fill.execution
        notional = exec_info.shares * exec_info.price

        record = {
            "timestamp": dt.datetime.now().isoformat(),
            "symbol": sym,
            "side": exec_info.side,
            "shares": exec_info.shares,
            "price": exec_info.price,
            "notional": notional,
            "orderId": exec_info.orderId,
            "execId": exec_info.execId,
            "commission": fill.commissionReport.commission if fill.commissionReport else 0.0,
        }
        self._fill_log.append(record)

        # Update risk counters
        self._risk.record_fill(notional)

        logger.info("FILL | {} {} {} @ ${:.2f} | notional=${:,.0f} | execId={}",
                     exec_info.side, exec_info.shares, sym, exec_info.price,
                     notional, exec_info.execId)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _count_positions(self) -> int:
        """Count current open positions."""
        try:
            return len(self._gw.ib.positions())
        except Exception:
            return 0

    def _position_notional(self, symbol: str) -> float:
        """Current notional exposure for a specific symbol."""
        try:
            for pos in self._gw.ib.positions():
                if pos.contract.symbol == symbol:
                    return abs(pos.position * pos.avgCost)
        except Exception:
            pass
        return 0.0
