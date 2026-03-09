"""Real-time position and PnL tracking from IB account.

Queries IB for current positions, calculates target vs actual weights,
and determines which trades are needed to rebalance.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger


class PortfolioTracker:
    """Track live positions and compute rebalance orders.

    Usage::

        pt = PortfolioTracker(gateway)
        positions = pt.positions()
        orders = pt.compute_rebalance(target_weights, capital=100_000)
    """

    def __init__(self, gateway):
        self._gw = gateway

    # ------------------------------------------------------------------
    # Position queries
    # ------------------------------------------------------------------

    def positions(self) -> pd.DataFrame:
        """Current positions as a DataFrame.

        Columns: symbol, quantity, avg_cost, market_value, unrealised_pnl.
        """
        ib = self._gw.ib
        pos_list = ib.positions()

        if not pos_list:
            return pd.DataFrame(columns=[
                "symbol", "quantity", "avg_cost", "market_value", "unrealised_pnl",
            ])

        rows = []
        for pos in pos_list:
            sym = pos.contract.symbol
            qty = pos.position
            avg = pos.avgCost
            mkt_val = qty * avg  # approximate; IB gives avgCost per share
            rows.append({
                "symbol": sym,
                "quantity": qty,
                "avg_cost": avg,
                "market_value": mkt_val,
                "unrealised_pnl": 0.0,  # updated via PnL subscription
            })

        return pd.DataFrame(rows)

    def position_dict(self) -> Dict[str, float]:
        """Return {symbol: quantity} for all positions."""
        df = self.positions()
        if df.empty:
            return {}
        return dict(zip(df["symbol"], df["quantity"]))

    def net_exposure(self) -> float:
        """Net dollar exposure (long - short)."""
        df = self.positions()
        if df.empty:
            return 0.0
        return (df["quantity"] * df["avg_cost"]).sum()

    def gross_exposure(self) -> float:
        """Gross dollar exposure (|long| + |short|)."""
        df = self.positions()
        if df.empty:
            return 0.0
        return (df["quantity"].abs() * df["avg_cost"]).sum()

    # ------------------------------------------------------------------
    # Rebalance computation
    # ------------------------------------------------------------------

    def compute_rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        capital: float,
        min_trade_notional: float = 500.0,
    ) -> List[Dict]:
        """Compute orders needed to move from current to target portfolio.

        Parameters
        ----------
        target_weights : {symbol: weight} where weights sum to ~1.0.
        prices : {symbol: current_price}.
        capital : total portfolio value to size positions.
        min_trade_notional : skip trades smaller than this (avoid dust).

        Returns
        -------
        List of order dicts: [{symbol, side, shares, price, notional}, ...]
        """
        current_pos = self.position_dict()

        orders = []
        all_symbols = set(list(target_weights.keys()) + list(current_pos.keys()))

        for sym in sorted(all_symbols):
            target_w = target_weights.get(sym, 0.0)
            price = prices.get(sym)

            if price is None or price <= 0:
                logger.warning("No price for {} — skipping", sym)
                continue

            target_shares = int(target_w * capital / price)
            current_shares = int(current_pos.get(sym, 0))
            delta = target_shares - current_shares

            if delta == 0:
                continue

            notional = abs(delta * price)
            if notional < min_trade_notional:
                continue

            side = "BUY" if delta > 0 else "SELL"
            orders.append({
                "symbol": sym,
                "side": side,
                "shares": abs(delta),
                "price": price,
                "notional": notional,
                "current_shares": current_shares,
                "target_shares": target_shares,
            })

        logger.info("Rebalance: {} orders ({} buys, {} sells)",
                     len(orders),
                     sum(1 for o in orders if o["side"] == "BUY"),
                     sum(1 for o in orders if o["side"] == "SELL"))

        return orders

    # ------------------------------------------------------------------
    # PnL
    # ------------------------------------------------------------------

    def pnl_summary(self) -> Dict:
        """Fetch account-level PnL from IB."""
        try:
            ib = self._gw.ib
            pnl_list = ib.pnl()
            if pnl_list:
                p = pnl_list[0]
                return {
                    "daily_pnl": p.dailyPnL,
                    "unrealised_pnl": p.unrealizedPnL,
                    "realised_pnl": p.realizedPnL,
                }
        except Exception as e:
            logger.debug("PnL query failed: {}", e)
        return {"daily_pnl": 0.0, "unrealised_pnl": 0.0, "realised_pnl": 0.0}

    def status_report(self) -> str:
        """Human-readable portfolio status."""
        pos = self.positions()
        pnl = self.pnl_summary()

        lines = [
            f"=== Portfolio Status @ {dt.datetime.now().strftime('%H:%M:%S')} ===",
            f"Positions: {len(pos)}",
            f"Net exposure:   ${self.net_exposure():>12,.0f}",
            f"Gross exposure: ${self.gross_exposure():>12,.0f}",
            f"Daily PnL:      ${pnl['daily_pnl'] or 0:>12,.2f}",
            f"Unrealised:     ${pnl['unrealised_pnl'] or 0:>12,.2f}",
            f"Realised:       ${pnl['realised_pnl'] or 0:>12,.2f}",
        ]

        if not pos.empty:
            lines.append("\nOpen Positions:")
            for _, row in pos.iterrows():
                lines.append(
                    f"  {row['symbol']:<6s}  qty={row['quantity']:>6.0f}  "
                    f"avg=${row['avg_cost']:>8.2f}"
                )

        return "\n".join(lines)
