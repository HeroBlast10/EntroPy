"""Spectral entropy and Hurst exponent factors wrapping TradeX Numba kernels.

Measure the noise structure of return series: spectral entropy quantifies
frequency-domain randomness, Hurst exponent captures long-range dependence.
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
def _rolling_spectral_entropy(returns, window):
    """Rolling spectral entropy via normalised periodogram.

    For each rolling window of *returns*, compute the power spectral density
    via squared DFT magnitudes, normalise to a probability distribution,
    and return the Shannon entropy divided by log(N) so the result lies in
    [0, 1] (1 = white noise, 0 = perfectly periodic).
    """
    n = len(returns)
    out = np.empty(n)
    out[:window - 1] = np.nan

    for i in range(window - 1, n):
        seg = returns[i - window + 1:i + 1]

        # Manual DFT (Numba-safe — no np.fft)
        N = len(seg)
        half = N // 2
        psd = np.empty(half)
        for k in range(half):
            re = 0.0
            im = 0.0
            for t in range(N):
                angle = 2.0 * np.pi * k * t / N
                re += seg[t] * np.cos(angle)
                im -= seg[t] * np.sin(angle)
            psd[k] = re * re + im * im

        # Normalise to probability distribution
        total = 0.0
        for k in range(half):
            total += psd[k]

        if total < 1e-30:
            out[i] = np.nan
            continue

        entropy = 0.0
        for k in range(half):
            p = psd[k] / total
            if p > 1e-30:
                entropy -= p * np.log(p)

        # Normalise by max entropy = log(half)
        max_ent = np.log(float(half))
        out[i] = entropy / max_ent if max_ent > 0.0 else np.nan

    return out


@njit(cache=True)
def _rolling_hurst(returns, window):
    """Rolling Hurst exponent via rescaled-range (R/S) analysis.

    Uses sub-intervals of sizes [8, 16, 32, ...] up to window//2
    and fits log(R/S) vs log(n) via OLS to obtain the Hurst exponent.
    """
    n = len(returns)
    out = np.empty(n)
    out[:window - 1] = np.nan

    for i in range(window - 1, n):
        seg = returns[i - window + 1:i + 1]
        W = len(seg)

        # Collect (log_n, log_rs) pairs for OLS
        max_pairs = 20
        log_ns = np.empty(max_pairs)
        log_rs = np.empty(max_pairs)
        count = 0

        size = 8
        while size <= W // 2 and count < max_pairs:
            num_blocks = W // size
            rs_sum = 0.0
            valid_blocks = 0

            for b in range(num_blocks):
                block = seg[b * size:(b + 1) * size]

                # Mean
                mean = 0.0
                for j in range(size):
                    mean += block[j]
                mean /= size

                # Cumulative deviation and std
                cum = 0.0
                cum_min = 1e30
                cum_max = -1e30
                ss = 0.0
                for j in range(size):
                    dev = block[j] - mean
                    cum += dev
                    if cum < cum_min:
                        cum_min = cum
                    if cum > cum_max:
                        cum_max = cum
                    ss += dev * dev

                std = np.sqrt(ss / size)
                if std > 1e-15:
                    rs_sum += (cum_max - cum_min) / std
                    valid_blocks += 1

            if valid_blocks > 0:
                log_ns[count] = np.log(float(size))
                log_rs[count] = np.log(rs_sum / valid_blocks)
                count += 1

            size *= 2

        if count < 2:
            out[i] = np.nan
            continue

        # OLS: slope of log(R/S) vs log(n)
        sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0
        for j in range(count):
            sx += log_ns[j]
            sy += log_rs[j]
            sxx += log_ns[j] * log_ns[j]
            sxy += log_ns[j] * log_rs[j]

        denom = count * sxx - sx * sx
        if abs(denom) < 1e-30:
            out[i] = np.nan
        else:
            out[i] = (count * sxy - sx * sy) / denom

    return out


# ===================================================================
# Helper
# ===================================================================

def _per_ticker_rolling(prices: pd.DataFrame, kernel, window: int) -> pd.Series:
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
        vals = kernel(rets, window)
        result[offset:offset + n] = vals
        offset += n

    return pd.Series(result, index=df.index).reindex(prices.index)


# ===================================================================
# Factor classes
# ===================================================================

class SpectralEntropy60D(FactorBase):
    meta = FactorMeta(
        name="SPECTRAL_ENTROPY_60D",
        category="noise_structure",
        signal_type="time_series",
        description="60-day spectral entropy of returns (1=pure noise, 0=periodic)",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        return _per_ticker_rolling(prices, _rolling_spectral_entropy, window=60)


class HurstExponent60D(FactorBase):
    meta = FactorMeta(
        name="HURST_60D",
        category="noise_structure",
        signal_type="time_series",
        description="60-day Hurst exponent via R/S analysis (>0.5=trending, <0.5=mean-reverting)",
        lookback=61,
        lag=1,
        direction=1,
    )

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.Series:
        return _per_ticker_rolling(prices, _rolling_hurst, window=60)


ALL_ENTROPY_HURST_FACTORS = [SpectralEntropy60D, HurstExponent60D]
