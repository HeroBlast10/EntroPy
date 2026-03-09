"""Value & Quality factors — placeholder until fundamentals pipeline is wired.

Factors
-------
1. **EARNINGS_YIELD**         — E/P ratio (trailing 12M earnings / market cap)
2. **GROSS_MARGIN_STABILITY** — 8-quarter rolling σ of gross margin (lower = higher quality)

Both require the fundamentals data pipeline (EPS, market cap, quarterly
income-statement fields).  They raise :class:`NotImplementedError` until
that pipeline is complete.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from quant_platform.core.signals.base import FactorBase, FactorMeta


class EarningsYield(FactorBase):
    meta = FactorMeta(
        name="EARNINGS_YIELD",
        category="value",
        signal_type="cross_sectional",
        description="Trailing 12M earnings / market cap (E/P ratio)",
        lookback=1,
        lag=1,
        direction=1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        raise NotImplementedError(
            "Earnings yield requires fundamentals data pipeline."
        )


class GrossMarginStability(FactorBase):
    meta = FactorMeta(
        name="GROSS_MARGIN_STABILITY",
        category="quality",
        signal_type="cross_sectional",
        description="8-quarter rolling std of gross margin (lower = higher quality)",
        lookback=1,
        lag=1,
        direction=-1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        raise NotImplementedError(
            "Gross margin stability requires fundamentals data pipeline."
        )


# Empty until fundamentals pipeline is complete
ALL_VALUE_QUALITY_FACTORS = []
