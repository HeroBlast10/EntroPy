"""Base class and common types for portfolio construction.

Every portfolio constructor inherits from :class:`PortfolioConstructor` and
implements :meth:`_generate_weights`.  The base class handles:

- Rebalance scheduling
- Constraint enforcement (post-processing)
- Weight normalisation
- Long-only / long-short mode switching
- Output formatting (date, ticker, weight)
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Enums & config types
# ===================================================================

class PortfolioMode(str, Enum):
    LONG_ONLY = "long_only"
    LONG_SHORT = "long_short"


class WeightScheme(str, Enum):
    EQUAL = "equal"
    MARKET_CAP = "market_cap"
    SIGNAL = "signal"            # weight ∝ signal strength
    INVERSE_VOL = "inverse_vol"


@dataclass
class PortfolioConfig:
    """Configuration shared by all portfolio constructors."""

    mode: PortfolioMode = PortfolioMode.LONG_ONLY
    weight_scheme: WeightScheme = WeightScheme.EQUAL

    # --- Position constraints ---
    max_stock_weight: float = 0.05          # 5 % per stock
    min_stock_weight: float = 0.0           # floor (0 = no floor)
    max_sector_weight: float = 0.30         # 30 % per sector
    max_turnover: Optional[float] = None    # max single-period turnover (None = unconstrained)

    # --- Selection ---
    n_quantiles: int = 5                    # for quantile-based methods
    long_quantile: int = 5                  # top quantile (buy)
    short_quantile: int = 1                 # bottom quantile (sell)
    top_n: Optional[int] = None             # if set, override quantile and pick top N

    # --- Rebalance ---
    rebalance_freq: str = "M"               # "D" / "W" / "M"

    # --- Misc ---
    initial_capital: float = 1_000_000.0


# ===================================================================
# Base class
# ===================================================================

class PortfolioConstructor(abc.ABC):
    """Abstract base class for all portfolio construction methods."""

    def __init__(self, config: Optional[PortfolioConfig] = None) -> None:
        self.config = config or PortfolioConfig()

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _generate_weights(
        self,
        signal: pd.DataFrame,
        universe: pd.DataFrame,
        date: pd.Timestamp,
        prev_weights: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Return target weights for a single rebalance date.

        Parameters
        ----------
        signal : full signal DataFrame ``[date, ticker, signal, …]``.
        universe : tradable universe for filtering.
        date : the rebalance date.
        prev_weights : weights from the previous period (for turnover control).

        Returns
        -------
        ``pd.Series`` indexed by ticker, values = target weight.
        Weights should sum to 1.0 (long-only) or have
        sum(long) = 1.0 and sum(short) = −1.0 (long-short).
        """
        ...

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        signal: pd.DataFrame,
        universe: pd.DataFrame,
        rebalance_dates: pd.DatetimeIndex,
        sector_map: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Run portfolio construction across all rebalance dates.

        Parameters
        ----------
        signal : ``[date, ticker, signal]`` — the alpha signal.
        universe : ``[date, ticker, pass_all_filters]`` from data layer.
        rebalance_dates : dates on which to rebalance.
        sector_map : ``[ticker, sector]`` for sector constraints.

        Returns
        -------
        DataFrame ``[date, ticker, weight]`` — one row per (date, holding).
        """
        signal = signal.copy()
        signal["date"] = pd.to_datetime(signal["date"])
        universe["date"] = pd.to_datetime(universe["date"])

        all_weights: List[pd.DataFrame] = []
        prev_weights: Optional[pd.Series] = None

        for dt in rebalance_dates:
            # Get today's signal & universe
            sig_today = signal.loc[signal["date"] == dt]
            uni_today = universe.loc[universe["date"] == dt]

            if sig_today.empty or uni_today.empty:
                continue

            # Filter to tradable universe
            tradable = set(uni_today["ticker"].unique())
            sig_today = sig_today[sig_today["ticker"].isin(tradable)]

            if sig_today.empty:
                continue

            # Generate raw weights
            raw_w = self._generate_weights(signal, universe, dt, prev_weights)

            if raw_w.empty:
                continue

            # Apply constraints
            from quant_platform.core.portfolio.constraints import apply_constraints
            final_w = apply_constraints(
                raw_w,
                config=self.config,
                sector_map=sector_map,
                prev_weights=prev_weights,
            )

            if final_w.empty:
                continue

            # Record
            wdf = pd.DataFrame({
                "date": dt,
                "ticker": final_w.index,
                "weight": final_w.values,
            })
            all_weights.append(wdf)
            prev_weights = final_w

        if not all_weights:
            logger.warning("No weights generated across any rebalance date")
            return pd.DataFrame(columns=["date", "ticker", "weight"])

        result = pd.concat(all_weights, ignore_index=True)
        logger.info(
            "Portfolio built: {} rebalance dates, {:.0f} avg holdings, mode={}",
            result["date"].nunique(),
            result.groupby("date").size().mean(),
            self.config.mode.value,
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def normalise_weights(
        weights: pd.Series,
        mode: PortfolioMode,
    ) -> pd.Series:
        """Normalise weights so they sum correctly.

        - Long-only: all weights ≥ 0, sum = 1.
        - Long-short: long side sums to +1, short side sums to −1.
        """
        if mode == PortfolioMode.LONG_ONLY:
            weights = weights.clip(lower=0.0)
            total = weights.sum()
            if total > 0:
                weights = weights / total
            return weights

        # Long-short
        longs = weights[weights > 0]
        shorts = weights[weights < 0]
        if longs.sum() > 0:
            longs = longs / longs.sum()
        if shorts.sum() < 0:
            shorts = shorts / shorts.abs().sum() * (-1)
        return pd.concat([longs, shorts])
