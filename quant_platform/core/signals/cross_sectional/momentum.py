"""Momentum & Reversal factors (multi-period).

Factors
-------
1. **MOM_1M**    — 1-month (21-day) momentum, skip 1 day
2. **MOM_3M**    — 3-month (63-day) momentum
3. **MOM_6M**    — 6-month (126-day) momentum
4. **MOM_12_1M** — 12-month momentum skipping the most recent month
                   (classic Jegadeesh & Titman 1993)
5. **STR_1W**    — 1-week short-term reversal
6. **STR_1M**    — 1-month short-term reversal (= −MOM_1M direction)
7. **MOM_PATH**  — Path-dependent momentum: cumulative return / max drawdown
                   during the lookback window (Daniel & Moskowitz 2016 flavour)

All momentum factors use **adj_close** to capture splits correctly.
The ``lag`` field on each factor ensures we never peek at the current day's
close when computing the signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quant_platform.core.signals.base import FactorBase, FactorMeta


# ===================================================================
# Helpers
# ===================================================================

def _pct_return(series: pd.Series, period: int) -> pd.Series:
    """Simple percentage return over *period* rows."""
    return series.pct_change(periods=period)


def _rolling_max_drawdown(series: pd.Series, window: int) -> pd.Series:
    """Rolling max drawdown over *window* rows."""
    rolling_max = series.rolling(window, min_periods=window).max()
    drawdown = series / rolling_max - 1.0
    return drawdown.rolling(window, min_periods=window).min()


# ===================================================================
# Factors
# ===================================================================

class Mom1M(FactorBase):
    meta = FactorMeta(
        name="MOM_1M",
        category="momentum",
        description="1-month (21-day) price momentum",
        lookback=22,
        lag=1,
        direction=1,
        references=["Jegadeesh & Titman (1993)"],
    )
    _DEFAULT_PERIOD = 21

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        df = prices.sort_values(["ticker", "date"])
        return df.groupby("ticker")["adj_close"].transform(lambda s: _pct_return(s, period))


class Mom3M(FactorBase):
    meta = FactorMeta(
        name="MOM_3M",
        category="momentum",
        description="3-month (63-day) price momentum",
        lookback=64,
        lag=1,
        direction=1,
        references=["Jegadeesh & Titman (1993)"],
    )
    _DEFAULT_PERIOD = 63

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        df = prices.sort_values(["ticker", "date"])
        return df.groupby("ticker")["adj_close"].transform(lambda s: _pct_return(s, period))


class Mom6M(FactorBase):
    meta = FactorMeta(
        name="MOM_6M",
        category="momentum",
        description="6-month (126-day) price momentum",
        lookback=127,
        lag=1,
        direction=1,
        references=["Jegadeesh & Titman (1993)"],
    )
    _DEFAULT_PERIOD = 126

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        df = prices.sort_values(["ticker", "date"])
        return df.groupby("ticker")["adj_close"].transform(lambda s: _pct_return(s, period))


class Mom12_1M(FactorBase):
    """Classic 12-minus-1-month momentum: skip the most recent 21 days."""

    meta = FactorMeta(
        name="MOM_12_1M",
        category="momentum",
        description="12-month momentum skipping last 1 month (Jegadeesh-Titman)",
        lookback=252,
        lag=21,  # skip the most recent month
        direction=1,
        references=["Jegadeesh & Titman (1993)"],
    )
    _DEFAULT_PERIOD = 252

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        df = prices.sort_values(["ticker", "date"])
        # period-day return; lag= in meta will shift to skip the most-recent month
        return df.groupby("ticker")["adj_close"].transform(lambda s: _pct_return(s, period))


class STR1W(FactorBase):
    """Short-term reversal: 1-week return (negative direction — lower is better
    because past losers tend to rebound)."""

    meta = FactorMeta(
        name="STR_1W",
        category="momentum",
        description="1-week (5-day) short-term reversal",
        lookback=6,
        lag=1,
        direction=-1,
        references=["Jegadeesh (1990)"],
    )
    _DEFAULT_PERIOD = 5

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        df = prices.sort_values(["ticker", "date"])
        return df.groupby("ticker")["adj_close"].transform(lambda s: _pct_return(s, period))


class STR1M(FactorBase):
    """Short-term reversal: 1-month return (negative direction)."""

    meta = FactorMeta(
        name="STR_1M",
        category="momentum",
        description="1-month (21-day) short-term reversal",
        lookback=22,
        lag=1,
        direction=-1,
        references=["Jegadeesh (1990)"],
    )
    _DEFAULT_PERIOD = 21

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        df = prices.sort_values(["ticker", "date"])
        return df.groupby("ticker")["adj_close"].transform(lambda s: _pct_return(s, period))


class MomPath(FactorBase):
    """Path-dependent momentum: cumulative return scaled by max drawdown.

    Captures the *quality* of the momentum path — smooth trends score
    higher than volatile ones with the same total return.  Inspired by
    Daniel & Moskowitz (2016) "Momentum Crashes".
    """

    meta = FactorMeta(
        name="MOM_PATH",
        category="momentum",
        description="Path-dependent momentum: return / |max drawdown| over 126 days",
        lookback=127,
        lag=1,
        direction=1,
        references=["Daniel & Moskowitz (2016)"],
    )

    _DEFAULT_WINDOW = 126

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        window = int(self._extra_params.get("window", self._DEFAULT_WINDOW))
        df = prices.sort_values(["ticker", "date"])

        def _path(s: pd.Series) -> pd.Series:
            ret = s.pct_change(periods=window)
            mdd = _rolling_max_drawdown(s, window).abs()
            mdd = mdd.replace(0, np.nan)
            return ret / mdd

        return df.groupby("ticker")["adj_close"].transform(_path)


# ===================================================================
# Convenience list
# ===================================================================

ALL_MOMENTUM_FACTORS = [
    Mom1M,
    Mom3M,
    Mom6M,
    Mom12_1M,
    STR1W,
    STR1M,
    MomPath,
]
