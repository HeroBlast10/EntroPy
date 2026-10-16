"""Transaction cost models.

Provides a pluggable cost framework with three components:

1. **Commission** — fixed per-share or percentage-of-notional broker fee.
2. **Slippage** — execution price deviation from the decision price,
   modelled as a fraction of the daily spread or volatility.
3. **Market impact** — price move caused by the trade itself, using a
   linear or square-root model of participation rate.

Additionally supports:
- **SEC fee** (US sell-side regulatory fee)
- **Stamp duty** (placeholder for non-US markets)
- **Financing cost** (for short positions held overnight)

All parameters have sensible US-equity defaults and are fully overridable.

Cost Model Hierarchy
--------------------
``CostModel`` is a dataclass that stores all parameters.
``estimate_trade_cost()`` is the stateless function that takes a single
trade and returns a cost breakdown dict.
``estimate_batch_costs()`` vectorises over a DataFrame of trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Cost model configuration
# ===================================================================

@dataclass
class CostModel:
    """All transaction-cost parameters in one place.

    Defaults are calibrated for US large-cap equities traded through a
    typical institutional broker (circa 2020–2024).
    """

    # --- Commission ---
    # Two modes: per-share or pct-of-notional (whichever is active)
    commission_per_share: float = 0.005     # $0.005 / share (Interactive Brokers tier)
    commission_pct: float = 0.0             # alternative: 0 if using per-share
    commission_min: float = 1.0             # minimum per order ($1)

    # --- Slippage ---
    # Modelled as a fraction of the daily VWAP spread.
    # slippage_bps: half-spread in basis points applied to notional.
    slippage_bps: float = 5.0               # 5 bps one-way (~1c on a $20 stock)

    # --- Market impact (linear model) ---
    # cost_impact = impact_coeff * sigma_daily * sqrt(participation_rate)
    # participation_rate = shares_traded / ADV
    # This is a simplified Almgren-Chriss / square-root model.
    impact_coeff: float = 0.1               # calibration constant
    impact_exponent: float = 0.5            # 0.5 = square-root model; 1.0 = linear

    # --- Regulatory fees (US) ---
    sec_fee_rate: float = 8.0e-6            # SEC fee: ~$8 per $1M (sells only)
    finra_taf_per_share: float = 0.000119   # FINRA TAF: $0.000119/share (sells only)
    finra_taf_max: Optional[float] = 8.30   # per-trade cap; set None to disable

    # --- Stamp duty (non-US placeholder) ---
    stamp_duty_pct: float = 0.0             # e.g. 0.001 for UK (10 bps, buys only)

    # --- Short-selling / financing ---
    borrow_rate_annual: float = 0.005       # 50 bps/year annualised borrow cost
    # Applied daily as borrow_rate_annual / 252 * abs(short_notional)

    # --- ADV lookback for participation rate ---
    adv_lookback: int = 20                  # trading days for average daily volume
    min_adv_shares: float = 1_000.0         # reject fills below this historical ADV
    max_participation_rate: Optional[float] = 0.10

    def __post_init__(self) -> None:
        nonnegative = (
            "commission_per_share",
            "commission_pct",
            "commission_min",
            "slippage_bps",
            "impact_coeff",
            "sec_fee_rate",
            "finra_taf_per_share",
            "stamp_duty_pct",
            "borrow_rate_annual",
            "min_adv_shares",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not np.isfinite(self.impact_exponent) or self.impact_exponent <= 0:
            raise ValueError("impact_exponent must be finite and positive")
        if self.finra_taf_max is not None and (
            not np.isfinite(self.finra_taf_max)
            or self.finra_taf_max < 0
        ):
            raise ValueError(
                "finra_taf_max must be finite and non-negative or None"
            )
        if int(self.adv_lookback) <= 0:
            raise ValueError("adv_lookback must be positive")
        if (
            self.max_participation_rate is not None
            and not 0 < self.max_participation_rate <= 1
        ):
            raise ValueError("max_participation_rate must be in (0, 1] or None")

    def describe(self) -> Dict[str, float]:
        """Return all parameters as a flat dict (useful for logging / serialisation)."""
        return {k: v for k, v in self.__dict__.items()}


# ===================================================================
# Single-trade cost estimation
# ===================================================================

def estimate_trade_cost(
    side: str,
    shares: float,
    price: float,
    adv_shares: float,
    daily_vol: float,
    model: CostModel,
) -> Dict[str, float]:
    """Estimate the all-in cost of a single trade.

    Parameters
    ----------
    side : ``"buy"`` or ``"sell"``
    shares : number of shares traded (always positive)
    price : execution / decision price ($)
    adv_shares : average daily volume in shares
    daily_vol : annualised daily return volatility (σ_daily, e.g. 0.02)
    model : cost model parameters

    Returns
    -------
    Dict with keys:
        ``notional``, ``commission``, ``slippage``, ``impact``,
        ``sec_fee``, ``finra_taf``, ``stamp_duty``, ``total_cost``,
        ``total_cost_bps``  (total as fraction of notional × 10 000)
    """
    shares = abs(shares)
    notional = shares * price

    if notional == 0 or shares == 0:
        return {k: 0.0 for k in (
            "notional", "commission", "slippage", "impact",
            "sec_fee", "finra_taf", "stamp_duty", "total_cost", "total_cost_bps",
        )}

    # 1. Commission
    if model.commission_pct > 0:
        comm = notional * model.commission_pct
    else:
        comm = shares * model.commission_per_share
    comm = max(comm, model.commission_min)

    # 2. Slippage (half-spread)
    slip = notional * (model.slippage_bps / 10_000)

    # 3. Market impact
    safe_adv = (
        float(adv_shares)
        if np.isfinite(adv_shares) and adv_shares > 0
        else max(shares, 1.0)
    )
    safe_daily_vol = (
        float(daily_vol)
        if np.isfinite(daily_vol) and daily_vol >= 0
        else 0.02
    )
    participation = shares / safe_adv
    # impact = coeff × σ_daily × participation^exponent × notional
    impact = (model.impact_coeff
              * safe_daily_vol
              * (participation ** model.impact_exponent)
              * notional)

    # 4. SEC fee (sells only)
    sec = notional * model.sec_fee_rate if side == "sell" else 0.0

    # 5. FINRA TAF (sells only)
    taf = shares * model.finra_taf_per_share if side == "sell" else 0.0
    if side == "sell" and model.finra_taf_max is not None:
        taf = min(taf, model.finra_taf_max)

    # 6. Stamp duty (buys only, non-US)
    stamp = notional * model.stamp_duty_pct if side == "buy" else 0.0

    total = comm + slip + impact + sec + taf + stamp
    bps = (total / notional * 10_000) if notional > 0 else 0.0

    return {
        "notional": round(notional, 2),
        "commission": round(comm, 4),
        "slippage": round(slip, 4),
        "impact": round(impact, 4),
        "sec_fee": round(sec, 4),
        "finra_taf": round(taf, 4),
        "stamp_duty": round(stamp, 4),
        "total_cost": round(total, 4),
        "total_cost_bps": round(bps, 2),
    }


# ===================================================================
# Batch cost estimation
# ===================================================================

def estimate_batch_costs(
    trades: pd.DataFrame,
    model: Optional[CostModel] = None,
) -> pd.DataFrame:
    """Estimate costs for a DataFrame of trades.

    Parameters
    ----------
    trades : must contain columns:
        ``side`` (buy/sell), ``shares``, ``price``,
        ``adv_shares``, ``daily_vol``
    model : cost model; uses defaults if ``None``.

    Returns
    -------
    *trades* with cost-breakdown columns appended.
    """
    if model is None:
        model = CostModel()

    cost_rows = []
    for _, row in trades.iterrows():
        c = estimate_trade_cost(
            side=row["side"],
            shares=row["shares"],
            price=row["price"],
            adv_shares=row.get("adv_shares", 1e6),
            daily_vol=row.get("daily_vol", 0.02),
            model=model,
        )
        cost_rows.append(c)

    costs_df = pd.DataFrame(cost_rows, index=trades.index)
    return pd.concat([trades, costs_df], axis=1)


# ===================================================================
# Overnight borrow cost (for short positions)
# ===================================================================

def daily_borrow_cost(
    short_notional: float,
    model: CostModel,
) -> float:
    """Daily borrow/financing cost for a short position.

    ``cost = |short_notional| × borrow_rate_annual / 252``
    """
    return abs(short_notional) * model.borrow_rate_annual / 252.0


# ===================================================================
# Cost summary helper
# ===================================================================

def summarise_costs(trades_with_costs: pd.DataFrame) -> Dict[str, float]:
    """Aggregate cost breakdown from a batch of trades."""
    cost_cols = ["commission", "slippage", "impact", "sec_fee", "finra_taf",
                 "stamp_duty", "total_cost"]
    total_notional = trades_with_costs["notional"].sum()
    summary = {}
    for col in cost_cols:
        if col in trades_with_costs.columns:
            summary[col] = trades_with_costs[col].sum()
    summary["total_notional"] = total_notional
    summary["total_cost_bps"] = (
        summary.get("total_cost", 0) / max(total_notional, 1) * 10_000
    )
    return summary
