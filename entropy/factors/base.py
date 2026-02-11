"""Abstract base class for all factors.

Every factor inherits from :class:`FactorBase` and implements:
- ``_compute(prices, fundamentals) → Series`` — the raw signal.
- Class-level metadata: ``name``, ``category``, ``lag``, ``lookback``.

The base class provides the full pipeline::

    raw signal → lag → winsorize → zscore → neutralize → final

so each concrete factor only has to define *what* to compute, not *how*
to post-process.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Metadata descriptor
# ===================================================================

@dataclass(frozen=True)
class FactorMeta:
    """Immutable metadata attached to every factor."""

    name: str                       # unique identifier, e.g. "MOM_12_1M"
    category: str                   # "momentum" | "volatility" | "liquidity"
    description: str                # one-liner for docs / tearsheet titles
    lookback: int                   # trading days of history required
    lag: int = 1                    # trading days to skip (prevent look-ahead)
    direction: int = 1             # +1 = higher is better, −1 = lower is better
    references: List[str] = field(default_factory=list)  # academic paper refs


# ===================================================================
# Base class
# ===================================================================

class FactorBase(abc.ABC):
    """Abstract base class for all alpha factors.

    Subclasses must set the class attribute ``meta`` (:class:`FactorMeta`)
    and implement :meth:`_compute`.
    """

    meta: FactorMeta  # must be set by subclass

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        """Return raw factor values indexed like *prices* ``(date, ticker)``.

        Must return a ``pd.Series`` with the same index as *prices*
        (or a subset).  The series name should be ``self.meta.name``.
        """
        ...

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
        *,
        winsorize_limits: Tuple[float, float] = (0.01, 0.99),
        zscore: bool = True,
        neutralize_by: Optional[List[str]] = None,
        fillna_method: str = "drop",
    ) -> pd.DataFrame:
        """Run the full factor pipeline: compute → lag → clean → transform.

        Parameters
        ----------
        prices : DataFrame with columns from the ``prices`` schema.
        fundamentals : optional fundamentals DataFrame.
        winsorize_limits : lower/upper quantile clipping bounds.
        zscore : if True, cross-sectional z-score each date.
        neutralize_by : columns in *prices* to neutralize against
            (e.g. ``["sector"]`` for industry neutralization, or
            ``["log_mcap"]`` for size, or both).
        fillna_method : ``"drop"`` | ``"zero"`` | ``"median"``.

        Returns
        -------
        DataFrame with columns ``[date, ticker, <factor_name>]``.
        """
        from entropy.factors.transforms import (
            apply_lag,
            cross_sectional_zscore,
            handle_missing,
            neutralize,
            winsorize,
        )

        name = self.meta.name
        logger.debug("Computing factor: {}", name)

        # 1. Raw signal
        raw = self._compute(prices, fundamentals)
        if isinstance(raw, pd.Series):
            raw.name = name

        # Build a working DataFrame
        if "date" in prices.columns and "ticker" in prices.columns:
            df = prices[["date", "ticker"]].copy()
            df[name] = raw.values if len(raw) == len(df) else raw.reindex(df.index).values
        else:
            df = raw.reset_index()
            df.columns = list(df.columns[:-1]) + [name]

        # 2. Lag
        if self.meta.lag > 0:
            df = apply_lag(df, name, lag=self.meta.lag)

        # 3. Missing values (first pass — before transforms)
        df = handle_missing(df, name, method=fillna_method)

        if df.empty:
            logger.warning("Factor {} produced an empty DataFrame after missing-value handling", name)
            return df

        # 4. Winsorize
        df = winsorize(df, name, limits=winsorize_limits)

        # 5. Z-score
        if zscore:
            df = cross_sectional_zscore(df, name)

        # 6. Neutralize
        if neutralize_by:
            df = neutralize(df, name, group_cols=neutralize_by)

        logger.info("Factor {}: {} rows, date range {} – {}",
                     name, len(df), df["date"].min(), df["date"].max())
        return df[["date", "ticker", name]]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        m = self.meta
        return f"<Factor {m.name} [{m.category}] lookback={m.lookback} lag={m.lag}>"
