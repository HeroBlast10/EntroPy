"""Higher-moment factors wrapping TradeX Numba kernels.

Rolling skewness, excess kurtosis, and autocorrelation decay exponent
capture the distributional shape and serial dependence structure of returns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from numba import njit

from quant_platform.core.signals.base import FactorBase, FactorMeta


# ===================================================================
# Numba kernels (ported from TradeX signal_factors.py)
# ===================================================================

@njit(cache=True)
def _rolling_skewness(returns, window):
    """Rolling Fisher skewness (third standardised moment)."""
    n = len(returns)
    out = np.empty(n)
    out[:window - 1] = np.nan

    for i in range(window - 1, n):
        # Mean
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += returns[j]
        mean /= window

        # Moments
        m2 = 0.0
        m3 = 0.0
        for j in range(i - window + 1, i + 1):
            d = returns[j] - mean
            d2 = d * d
            m2 += d2
            m3 += d2 * d
        m2 /= window
        m3 /= window

        if m2 < 1e-30:
            out[i] = np.nan
        else:
            std = np.sqrt(m2)
            out[i] = m3 / (std * std * std)

    return out


@njit(cache=True)
def _rolling_kurtosis(returns, window):
    """Rolling excess kurtosis (fourth standardised moment minus 3)."""
    n = len(returns)
    out = np.empty(n)
    out[:window - 1] = np.nan

    for i in range(window - 1, n):
        # Mean
        mean = 0.0
        for j in range(i - window + 1, i + 1):
            mean += returns[j]
        mean /= window

        # Moments
        m2 = 0.0
        m4 = 0.0
        for j in range(i - window + 1, i + 1):
            d = returns[j] - mean
            d2 = d * d
            m2 += d2
            m4 += d2 * d2
        m2 /= window
        m4 /= window

        if m2 < 1e-30:
            out[i] = np.nan
        else:
            out[i] = m4 / (m2 * m2) - 3.0

    return out


@njit(cache=True)
def _rolling_acf_decay(returns, window, max_lag=10):
    """Rolling ACF power-law decay exponent.

    Computes sample ACF at lags 1..max_lag, then fits log(|ACF|) vs log(lag)
    via OLS. The slope is the decay exponent (larger magnitude = faster decay).
    """
    n = len(returns)
    out = np.empty(n)
    out[:window - 1] = np.nan

    for i in range(window - 1, n):
        seg = returns[i - window + 1:i + 1]
        W = len(seg)

        # Mean & variance
        mean = 0.0
        for j in range(W):
            mean += seg[j]
        mean /= W

        var = 0.0
        for j in range(W):
            d = seg[j] - mean
            var += d * d
        var /= W

        if var < 1e-30:
            out[i] = np.nan
            continue

        # ACF at lags 1..max_lag
        num_lags = min(max_lag, W // 4)
        if num_lags < 2:
            out[i] = np.nan
            continue

        log_lag = np.empty(num_lags)
        log_acf = np.empty(num_lags)
        valid = 0

        for lag in range(1, num_lags + 1):
            acf_val = 0.0
            for j in range(lag, W):
                acf_val += (seg[j] - mean) * (seg[j - lag] - mean)
            acf_val /= (W * var)

            if abs(acf_val) > 1e-15:
                log_lag[valid] = np.log(float(lag))
                log_acf[valid] = np.log(abs(acf_val))
                valid += 1

        if valid < 2:
            out[i] = np.nan
            continue

        # OLS slope
        sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
        for j in range(valid):
            sx += log_lag[j]
            sy += log_acf[j]
            sxx += log_lag[j] * log_lag[j]
            sxy += log_lag[j] * log_acf[j]

        denom = valid * sxx - sx * sx
        if abs(denom) < 1e-30:
            out[i] = np.nan
        else:
            out[i] = (valid * sxy - sx * sy) / denom

    return out


# ===================================================================
# Helper
# ===================================================================

def _per_ticker_rolling(prices: pd.DataFrame, kernel, window: int, **kwargs) -> pd.Series:
    """Apply a rolling Numba kernel to daily returns for each ticker."""
    df = prices.sort_values(["ticker", "date"]).copy()
    result = np.full(len(df), np.nan)

    offset = 0
    for _ticker, grp in df.groupby("ticker", sort=False):
        n = len(grp)
        adj = grp["adj_close"].values.astype(np.float64)
        rets = np.empty(n)
        rets[0] = 0.0
        for j in range(1, n):
            rets[j] = adj[j] / adj[j - 1] - 1.0 if adj[j - 1] != 0.0 else 0.0
        vals = kernel(rets, window, **kwargs)
        result[offset:offset + n] = vals
        offset += n

    return pd.Series(result, index=df.index).reindex(prices.index)


# ===================================================================
# Factor classes
# ===================================================================

class RollingSkew60D(FactorBase):
    meta = FactorMeta(
        name="ROLLING_SKEW_60D",
        category="higher_moments",
        signal_type="time_series",
        description="60-day rolling skewness of returns (Fisher definition)",
        lookback=61,
        lag=1,
        direction=1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        return _per_ticker_rolling(prices, _rolling_skewness, window=60)


class RollingKurt60D(FactorBase):
    meta = FactorMeta(
        name="ROLLING_KURT_60D",
        category="higher_moments",
        signal_type="time_series",
        description="60-day rolling excess kurtosis (fat tails indicator)",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        return _per_ticker_rolling(prices, _rolling_kurtosis, window=60)


class ACFDecay60D(FactorBase):
    meta = FactorMeta(
        name="ACF_DECAY_60D",
        category="higher_moments",
        signal_type="time_series",
        description="60-day ACF power-law decay exponent (larger=faster decay=mean-reverting)",
        lookback=61,
        lag=1,
        direction=1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        return _per_ticker_rolling(prices, _rolling_acf_decay, window=60)


ALL_HIGHER_MOMENT_FACTORS = [RollingSkew60D, RollingKurt60D, ACFDecay60D]
