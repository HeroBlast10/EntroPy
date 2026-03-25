"""Volatility, Risk & Tail factors.

Incorporates a stochastic-process perspective: realized vol, jump
detection, volatility-of-volatility, and tail-risk measures.

Factors
-------
1.  **VOL_20D**      — 20-day realized volatility (annualized)
2.  **VOL_60D**      — 60-day realized volatility
3.  **IDIOVOL**      — Idiosyncratic volatility: residual std from
                       CAPM regression over 60 days
4.  **SKEW_60D**     — 60-day return skewness (negative skew = crash-prone)
5.  **KURT_60D**     — 60-day return kurtosis (heavy tails)
6.  **DOWNVOL_60D**  — Downside semi-deviation over 60 days
7.  **TAIL_RISK**    — CVaR (Expected Shortfall) at 5 % over 60 days
8.  **VOL_OF_VOL**   — Volatility of 20-day realized vol measured over
                       60 days (stochastic-vol ν parameter proxy)
9.  **REALIZED_JUMP** — Bi-power variation jump ratio: fraction of
                        variance attributable to jumps (simplified
                        Barndorff-Nielsen & Shephard 2004)

All factors use **adj_close** daily returns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    from numba import jit
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        """Dummy decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])

from quant_platform.core.signals.base import FactorBase, FactorMeta


# ===================================================================
# Helpers
# ===================================================================

_ANN = np.sqrt(252)


def _daily_returns(group: pd.Series) -> pd.Series:
    return group.pct_change()


@jit(nopython=True, cache=True)
def _ols_residual_std(y: np.ndarray, x: np.ndarray) -> float:
    """OLS residual standard deviation (single regressor)."""
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 10:
        return np.nan
    xm, ym = x[mask], y[mask]
    xm_dm = xm - xm.mean()
    denom = (xm_dm ** 2).sum()
    if denom == 0:
        return np.nan
    beta = (xm_dm * (ym - ym.mean())).sum() / denom
    alpha = ym.mean() - beta * xm.mean()
    resid = ym - (alpha + beta * xm)
    return float(np.std(resid, ddof=1))


@jit(nopython=True, cache=True)
def _rolling_downside_std(arr: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    """Fast rolling downside semi-deviation (only negative values).
    
    Parameters
    ----------
    arr : 1D array of returns
    window : rolling window size
    min_periods : minimum observations required
    
    Returns
    -------
    Array of rolling downside std (same length as input)
    """
    n = len(arr)
    out = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        w = arr[i - window + 1:i + 1]
        # Filter to negative returns only
        neg = w[w < 0]
        if len(neg) >= min_periods:
            out[i] = np.sqrt(np.mean(neg ** 2))
        elif len(w[np.isfinite(w)]) >= min_periods:
            out[i] = 0.0  # No negative returns in window
    
    return out


@jit(nopython=True, cache=True)
def _rolling_cvar(arr: np.ndarray, window: int, min_periods: int, quantile: float = 0.05) -> np.ndarray:
    """Fast rolling CVaR (Expected Shortfall).
    
    Parameters
    ----------
    arr : 1D array of returns
    window : rolling window size
    min_periods : minimum observations required
    quantile : tail quantile (default 5%)
    
    Returns
    -------
    Array of rolling CVaR (same length as input)
    """
    n = len(arr)
    out = np.full(n, np.nan)
    
    for i in range(window - 1, n):
        w = arr[i - window + 1:i + 1]
        valid = w[np.isfinite(w)]
        if len(valid) >= min_periods:
            threshold = np.quantile(valid, quantile)
            tail = valid[valid <= threshold]
            if len(tail) > 0:
                out[i] = np.mean(tail)
    
    return out


@jit(nopython=True, cache=True)
def _rolling_idiovol(ret: np.ndarray, mkt_ret: np.ndarray, window: int) -> np.ndarray:
    """Fast rolling idiosyncratic volatility via CAPM.
    
    Parameters
    ----------
    ret : 1D array of stock returns
    mkt_ret : 1D array of market returns (aligned)
    window : rolling window size
    
    Returns
    -------
    Array of rolling idiosyncratic vol (annualized)
    """
    n = len(ret)
    out = np.full(n, np.nan)
    
    for i in range(window, n):
        y_w = ret[i - window:i]
        x_w = mkt_ret[i - window:i]
        out[i] = _ols_residual_std(y_w, x_w)
    
    return out * np.sqrt(252.0)


# ===================================================================
# Factors
# ===================================================================

class Vol20D(FactorBase):
    meta = FactorMeta(
        name="VOL_20D",
        category="volatility",
        description="20-day annualized realized volatility",
        lookback=21,
        lag=1,
        direction=-1,
        references=["Ang, Hodrick, Xing & Zhang (2006)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        vol = ret.groupby(df["ticker"]).transform(
            lambda s: s.rolling(20, min_periods=15).std() * _ANN
        )
        return vol


class Vol60D(FactorBase):
    meta = FactorMeta(
        name="VOL_60D",
        category="volatility",
        description="60-day annualized realized volatility",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        vol = ret.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).std() * _ANN
        )
        return vol


class IdioVol(FactorBase):
    """Idiosyncratic volatility — residual std from a rolling CAPM regression.

    We use equal-weighted market return as the benchmark.  This measures
    the stock-specific risk that cannot be explained by broad market moves.
    """

    meta = FactorMeta(
        name="IDIOVOL",
        category="volatility",
        description="60-day idiosyncratic volatility (CAPM residual std)",
        lookback=61,
        lag=1,
        direction=-1,
        references=["Ang, Hodrick, Xing & Zhang (2006)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"]).copy()
        df["ret"] = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        # Market return: equal-weighted average across all tickers each day
        mkt = df.groupby("date")["ret"].mean().rename("mkt_ret")
        df = df.merge(mkt, on="date", how="left")

        # Fast vectorized computation using Numba
        def _compute_idiovol(group: pd.DataFrame) -> pd.Series:
            ret_arr = group["ret"].values
            mkt_arr = group["mkt_ret"].values
            result = _rolling_idiovol(ret_arr, mkt_arr, window=60)
            return pd.Series(result, index=group.index)

        if "ticker" not in df.columns:
            return pd.Series(dtype="float64")
        
        return df.groupby("ticker", group_keys=False).apply(_compute_idiovol)


class Skew60D(FactorBase):
    meta = FactorMeta(
        name="SKEW_60D",
        category="volatility",
        description="60-day return skewness (negative = crash-prone)",
        lookback=61,
        lag=1,
        direction=1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        return ret.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).skew()
        )


class Kurt60D(FactorBase):
    meta = FactorMeta(
        name="KURT_60D",
        category="volatility",
        description="60-day return excess kurtosis (heavy tails)",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        return ret.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).kurt()
        )


class DownVol60D(FactorBase):
    """Downside semi-deviation: volatility computed only on negative returns."""

    meta = FactorMeta(
        name="DOWNVOL_60D",
        category="volatility",
        description="60-day downside semi-deviation (annualized)",
        lookback=61,
        lag=1,
        direction=-1,
        references=["Sortino & van der Meer (1991)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)

        # Fast vectorized computation using Numba
        def _compute_downvol(group: pd.Series) -> pd.Series:
            arr = group.values
            result = _rolling_downside_std(arr, window=60, min_periods=40)
            return pd.Series(result * _ANN, index=group.index)

        return ret.groupby(df["ticker"]).transform(_compute_downvol)


class TailRisk(FactorBase):
    """Conditional Value-at-Risk (CVaR / Expected Shortfall) at 5 % tail."""

    meta = FactorMeta(
        name="TAIL_RISK",
        category="volatility",
        description="60-day CVaR at 5% (Expected Shortfall)",
        lookback=61,
        lag=1,
        direction=-1,
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)

        # Fast vectorized computation using Numba
        def _compute_cvar(group: pd.Series) -> pd.Series:
            arr = group.values
            result = _rolling_cvar(arr, window=60, min_periods=40, quantile=0.05)
            return pd.Series(result, index=group.index)

        return ret.groupby(df["ticker"]).transform(_compute_cvar)


class VolOfVol(FactorBase):
    """Volatility of volatility — proxy for the ν (vol-of-vol) parameter
    in stochastic volatility models (e.g. Heston 1993).

    Computed as the rolling std of 20-day realized vol over 60 days.
    High vol-of-vol stocks exhibit regime-switching behaviour and
    are harder to hedge.
    """

    meta = FactorMeta(
        name="VOL_OF_VOL",
        category="volatility",
        description="Volatility of 20-day realized vol over 60 days (stochastic-vol proxy)",
        lookback=80,
        lag=1,
        direction=-1,
        references=["Heston (1993)", "Baltussen, van Bekkum & Grient (2018)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        # Step 1: 20-day rolling vol
        vol20 = ret.groupby(df["ticker"]).transform(
            lambda s: s.rolling(20, min_periods=15).std() * _ANN
        )
        # Step 2: 60-day rolling std of that vol
        return vol20.groupby(df["ticker"]).transform(
            lambda s: s.rolling(60, min_periods=40).std()
        )


class RealizedJump(FactorBase):
    """Realized jump variation ratio (simplified Barndorff-Nielsen & Shephard).

    Jump ratio = 1 − BPV / RV, where:
    - RV  = sum of squared returns (realized variance)
    - BPV = (π/2) × sum of |r_t| × |r_{t−1}| (bi-power variation,
            estimates the continuous component)

    Higher ratio → more of the variance comes from jumps → more
    discontinuous / event-driven price process.
    """

    meta = FactorMeta(
        name="REALIZED_JUMP",
        category="volatility",
        description="60-day realized jump ratio (BPV-based, Barndorff-Nielsen & Shephard)",
        lookback=61,
        lag=1,
        direction=-1,
        references=["Barndorff-Nielsen & Shephard (2004)"],
    )

    def _compute(self, prices: pd.DataFrame, fundamentals=None) -> pd.Series:
        df = prices.sort_values(["ticker", "date"])
        ret = df.groupby("ticker")["adj_close"].transform(_daily_returns)
        abs_ret = ret.abs()
        abs_ret_lag = abs_ret.groupby(df["ticker"]).shift(1)
        bpv_term = abs_ret * abs_ret_lag  # |r_t| * |r_{t-1}|

        window = 60

        def _jump_ratio(idx):
            rv = (ret.loc[idx] ** 2).rolling(window, min_periods=40).sum()
            bpv = (np.pi / 2) * bpv_term.loc[idx].rolling(window, min_periods=40).sum()
            ratio = 1.0 - bpv / rv.replace(0, np.nan)
            return ratio.clip(lower=0.0)  # ratio ∈ [0, 1]

        return df.groupby("ticker").apply(
            lambda g: _jump_ratio(g.index), include_groups=False,
        ).droplevel(0).sort_index()


# ===================================================================
# Convenience list
# ===================================================================

ALL_VOLATILITY_FACTORS = [
    Vol20D,
    Vol60D,
    IdioVol,
    Skew60D,
    Kurt60D,
    DownVol60D,
    TailRisk,
    VolOfVol,
    RealizedJump,
]
