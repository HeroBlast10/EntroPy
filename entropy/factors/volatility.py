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

from entropy.factors.base import FactorBase, FactorMeta


# ===================================================================
# Helpers
# ===================================================================

_ANN = np.sqrt(252)


def _daily_returns(group: pd.Series) -> pd.Series:
    return group.pct_change()


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

        window = 60

        def _idio(sub: pd.DataFrame) -> pd.Series:
            out = pd.Series(np.nan, index=sub.index)
            y_arr = sub["ret"].values
            x_arr = sub["mkt_ret"].values
            for i in range(window, len(sub)):
                y_w = y_arr[i - window:i]
                x_w = x_arr[i - window:i]
                out.iloc[i] = _ols_residual_std(y_w, x_w)
            return out * _ANN

        return df.groupby("ticker", group_keys=False).apply(_idio).droplevel(0) \
            if "ticker" in df.columns else pd.Series(dtype="float64")


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
        neg_ret = ret.where(ret < 0, 0.0)

        def _semi_std(s: pd.Series) -> pd.Series:
            return s.rolling(60, min_periods=40).apply(
                lambda w: np.sqrt((w[w < 0] ** 2).mean()) if (w < 0).any() else 0.0,
                raw=True,
            ) * _ANN

        return neg_ret.groupby(df["ticker"]).transform(_semi_std)


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

        def _cvar(s: pd.Series) -> pd.Series:
            return s.rolling(60, min_periods=40).apply(
                lambda w: w[w <= np.nanquantile(w, 0.05)].mean()
                if len(w[np.isfinite(w)]) >= 10 else np.nan,
                raw=True,
            )

        return ret.groupby(df["ticker"]).transform(_cvar)


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
