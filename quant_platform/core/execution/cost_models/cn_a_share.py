"""A-share transaction cost model.

Migrated from TradeX. Realistic cost simulation for CN A-share market:
- Commission (bilateral, ~3 bps, min 5 CNY)
- Stamp tax (sell-only, 5 bps post-Aug 2023)
- Transfer fee (bilateral, ~0.1 bps)
- Slippage (fixed / volume-based / volatility-based)

Operates on weight-change matrices for fully vectorized computation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AShareCostModel:
    """A-share transaction cost model.

    All rates are in decimal (e.g., 0.0003 = 0.03% = 3 bps).
    """

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.0005
    transfer_fee_rate: float = 0.00001
    slippage_model: str = "fixed"
    slippage_bps: float = 5.0
    slippage_k: float = 0.1
    min_commission: float = 5.0

    def compute_cost_matrix(
        self,
        weight_prev: np.ndarray,
        weight_new: np.ndarray,
        close_prices: np.ndarray,
        nav: float = 1e6,
        adv: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute per-stock transaction cost for a single rebalance.

        Returns cost as fraction of NAV (not in CNY).
        """
        delta_w = weight_new - weight_prev
        abs_delta = np.abs(delta_w)
        trade_value = abs_delta * nav

        commission = abs_delta * self.commission_rate
        sell_mask = (delta_w < 0).astype(np.float64)
        stamp = np.abs(delta_w) * sell_mask * self.stamp_tax_rate
        transfer = abs_delta * self.transfer_fee_rate
        slippage = self._compute_slippage(abs_delta, trade_value, close_prices, adv, volatility)

        return commission + stamp + transfer + slippage

    def compute_total_cost(
        self,
        weight_prev: np.ndarray,
        weight_new: np.ndarray,
        nav: float = 1e6,
        close_prices: Optional[np.ndarray] = None,
        adv: Optional[np.ndarray] = None,
        volatility: Optional[np.ndarray] = None,
    ) -> float:
        if close_prices is None:
            close_prices = np.ones(len(weight_prev))
        cost_per_stock = self.compute_cost_matrix(
            weight_prev, weight_new, close_prices, nav, adv, volatility,
        )
        return float(np.nansum(cost_per_stock))

    def _compute_slippage(self, abs_delta, trade_value, close_prices, adv, volatility):
        if self.slippage_model == "fixed":
            return abs_delta * (self.slippage_bps / 10000.0)
        elif self.slippage_model == "volume" and adv is not None:
            adv_value = adv * close_prices
            adv_value = np.where(adv_value > 0, adv_value, 1e10)
            participation = trade_value / adv_value
            return self.slippage_k * np.sqrt(np.clip(participation, 0, 1)) * abs_delta
        elif self.slippage_model == "volatility" and volatility is not None:
            return self.slippage_k * volatility * np.sqrt(abs_delta)
        return abs_delta * (self.slippage_bps / 10000.0)

    def summary(self) -> str:
        return (
            f"AShareCostModel(\n"
            f"  commission={self.commission_rate * 10000:.1f}bps bilateral,\n"
            f"  stamp_tax={self.stamp_tax_rate * 10000:.1f}bps sell-only,\n"
            f"  transfer={self.transfer_fee_rate * 10000:.2f}bps bilateral,\n"
            f"  slippage={self.slippage_model}"
            f"{'@' + str(self.slippage_bps) + 'bps' if self.slippage_model == 'fixed' else ''}\n"
            f")"
        )
