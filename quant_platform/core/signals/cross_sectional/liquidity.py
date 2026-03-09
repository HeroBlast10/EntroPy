"""Liquidity, Volume & Turnover factors.

Factors
-------
1.  **TURNOVER_20D**      — 20-day average daily turnover (volume / shares out proxy)
2.  **TURNOVER_60D**      — 60-day average daily turnover
3.  **ILLIQ_AMIHUD**      — Amihud (2002) illiquidity ratio: mean(|ret| / dollar volume)
4.  **VOLUME_CV**         — Coefficient of variation of daily volume over 60 days
5.  **PRICE_IMPACT**      — Kyle's lambda proxy: regress |ret| on signed volume
6.  **TURNOVER_ACCEL**    — Turnover acceleration: change in 20-day turnover vs 60-day
7.  **ABNORMAL_VOLUME**   — Volume z-score: today's volume vs 60-day trailing mean/std
8.  **SPREAD_HL**         — High-low spread estimator (Corwin & Schultz 2012)

All factors use **adj_close**, **volume**, **high**, **low** from the
prices table.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from quant_platform.core.signals.base import FactorBase, FactorMeta


# ===================================================================
# Helpers
# ===================================================================

def _safe_div(a, b):
    """Element-wise a/b, returning NaN where b is 0 or NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = a / b
    return result


# ===================================================================
# Factors
# ===================================================================

class Turnover20D(FactorBase):
    """20-day average daily turnover.

    Since shares-outstanding is not always available, we proxy turnover as
    ``volume / 60-day-median(volume)`` — a *relative* turnover measure
    that is comparable across stocks.
    """

    meta = FactorMeta(
        name="TURNOVER_20D",
        category="liquidity",
        description="20-day average relative daily turnover",
        lookback=21,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        vol = df["volume"].astype(float)
        # Relative turnover: volume / trailing 60-day median volume
        median_vol = vol.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=20).median()
        )
        rel_turn = _safe_div(vol, median_vol)
        # 20-day average
        return rel_turn.groupby(df["ticker"]).transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )


class Turnover60D(FactorBase):
    meta = FactorMeta(
        name="TURNOVER_60D",
        category="liquidity",
        description="60-day average relative daily turnover",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        vol = df["volume"].astype(float)
        median_vol = vol.groupby(df["ticker"]).transform(
            lambda s: s.rolling(120, min_periods=40).median()
        )
        rel_turn = _safe_div(vol, median_vol)
        return rel_turn.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).mean()
        )


class IlliqAmihud(FactorBase):
    """Amihud (2002) illiquidity ratio.

    ILLIQ = mean(|daily return| / dollar volume) over a rolling window.
    Higher = more illiquid (price moves a lot per dollar traded).
    """

    meta = FactorMeta(
        name="ILLIQ_AMIHUD",
        category="liquidity",
        description="Amihud illiquidity: mean(|ret|/dollar_volume) over 60 days",
        lookback=61,
        lag=1,
        direction=-1,
        references=["Amihud (2002)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(lambda s: s.pct_change())
        dollar_vol = df["close"].abs() * df["volume"].astype(float)
        # |ret| / dollar_volume, scaled by 1e6 to keep numbers reasonable
        illiq_daily = _safe_div(ret.abs(), dollar_vol) * 1e6
        return illiq_daily.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).mean()
        )


class VolumeCV(FactorBase):
    """Coefficient of variation of daily volume over 60 days.

    High CV indicates erratic trading activity — potentially driven by
    information events or speculative bursts.
    """

    meta = FactorMeta(
        name="VOLUME_CV",
        category="liquidity",
        description="60-day coefficient of variation of daily volume",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        vol = df["volume"].astype(float)

        def _cv(s: pd.Series) -> pd.Series:
            mu = s.rolling(60, min_periods=40).mean()
            sigma = s.rolling(60, min_periods=40).std()
            return _safe_div(sigma, mu)

        return vol.groupby(df["ticker"]).transform(_cv)


class PriceImpact(FactorBase):
    """Kyle's lambda proxy — price impact per unit of signed volume.

    For each 60-day window we regress |return| on signed_volume and take
    the slope.  Higher slope = larger price impact = less liquid.

    Simplified: we use correlation(|ret|, signed_vol) × std(|ret|) / std(vol)
    as a rolling beta estimate to avoid per-window OLS.
    """

    meta = FactorMeta(
        name="PRICE_IMPACT",
        category="liquidity",
        description="Kyle's lambda proxy: rolling beta of |ret| on signed volume (60d)",
        lookback=61,
        lag=1,
        direction=-1,
        references=["Kyle (1985)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"]).copy()
        ret = df.groupby("ticker")["adj_close"].transform(lambda s: s.pct_change())
        # Signed volume: volume × sign(return)
        sign_vol = df["volume"].astype(float) * np.sign(ret)

        window = 60

        def _rolling_beta(sub: pd.DataFrame) -> pd.Series:
            x = sub["_svol"]
            y = sub["_absret"]
            cov_xy = x.rolling(window, min_periods=40).cov(y)
            var_x = x.rolling(window, min_periods=40).var()
            return _safe_div(cov_xy, var_x)

        df["_absret"] = ret.abs()
        df["_svol"] = sign_vol
        result = df.groupby("ticker", group_keys=False).apply(_rolling_beta)
        return result


class TurnoverAccel(FactorBase):
    """Turnover acceleration: short-term turnover vs long-term turnover.

    Ratio of 20-day mean volume to 60-day mean volume minus 1.
    Positive → recent volume surge; Negative → volume contraction.
    """

    meta = FactorMeta(
        name="TURNOVER_ACCEL",
        category="liquidity",
        description="Turnover acceleration: (20d avg vol / 60d avg vol) − 1",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        vol = df["volume"].astype(float)
        ma20 = vol.groupby(df["ticker"]).transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )
        ma60 = vol.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).mean()
        )
        return _safe_div(ma20, ma60) - 1.0


class AbnormalVolume(FactorBase):
    """Abnormal volume z-score: how many std's today's volume deviates from
    the trailing 60-day distribution."""

    meta = FactorMeta(
        name="ABNORMAL_VOLUME",
        category="liquidity",
        description="Volume z-score vs 60-day trailing mean/std",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        vol = df["volume"].astype(float)
        mu = vol.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).mean()
        )
        sigma = vol.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).std()
        )
        return _safe_div(vol - mu, sigma)


class SpreadHL(FactorBase):
    """High-Low spread estimator (Corwin & Schultz 2012).

    Decomposes the daily high-low range into a volatility component and
    a bid-ask spread component.  Uses a 2-day estimator averaged over
    a 20-day window.  Higher = wider spread = less liquid.
    """

    meta = FactorMeta(
        name="SPREAD_HL",
        category="liquidity",
        description="Corwin-Schultz high-low spread estimator (20-day average)",
        lookback=22,
        lag=1,
        direction=-1,
        references=["Corwin & Schultz (2012)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"]).copy()
        # β = ln(H_t/L_t)^2 — single-day range
        log_hl = np.log(df["high"] / df["low"].replace(0, np.nan))
        beta = log_hl ** 2

        # γ = ln(max(H_t, H_{t-1}) / min(L_t, L_{t-1}))^2  — 2-day range
        h2 = df.groupby("ticker")["high"].transform(
            lambda s: s.rolling(2, min_periods=2).max()
        )
        l2 = df.groupby("ticker")["low"].transform(
            lambda s: s.rolling(2, min_periods=2).min()
        )
        gamma = np.log(h2 / l2.replace(0, np.nan)) ** 2

        # Corwin-Schultz formula
        beta_sum = beta.groupby(df["ticker"]).transform(
            lambda s: s.rolling(2, min_periods=2).sum()
        )
        k = 2 * np.sqrt(2) - 1
        alpha = (_safe_div(np.sqrt(2 * beta_sum) - np.sqrt(beta_sum), k)
                 - np.sqrt(_safe_div(gamma, k)))
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        spread = spread.clip(lower=0.0)

        # Average over 20 days
        return spread.groupby(df["ticker"]).transform(
            lambda s: s.rolling(20, min_periods=15).mean()
        )


# ===================================================================
# Convenience list
# ===================================================================

ALL_LIQUIDITY_FACTORS = [
    Turnover20D,
    Turnover60D,
    IlliqAmihud,
    VolumeCV,
    PriceImpact,
    TurnoverAccel,
    AbnormalVolume,
    SpreadHL,
]
