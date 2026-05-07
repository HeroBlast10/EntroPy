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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        if _feature_cache is not None:
            return _feature_cache.get(f"ret_{period}d")
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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        if _feature_cache is not None:
            return _feature_cache.get(f"ret_{period}d")
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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        if _feature_cache is not None:
            return _feature_cache.get(f"ret_{period}d")
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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        if _feature_cache is not None:
            return _feature_cache.get(f"ret_{period}d")
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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        if _feature_cache is not None:
            return _feature_cache.get(f"ret_{period}d")
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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        period = int(self._extra_params.get("period", self._DEFAULT_PERIOD))
        if _feature_cache is not None:
            return _feature_cache.get(f"ret_{period}d")
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

    def _compute(self, prices: pd.DataFrame, fundamentals=None, _feature_cache=None) -> pd.Series:
        window = int(self._extra_params.get("window", self._DEFAULT_WINDOW))
        df = prices.sort_values(["ticker", "date"])

        def _path(s: pd.Series) -> pd.Series:
            ret = s.pct_change(periods=window)
            mdd = _rolling_max_drawdown(s, window).abs()
            mdd = mdd.replace(0, np.nan)
            return ret / mdd

        return df.groupby("ticker")["adj_close"].transform(_path)


# ===================================================================
# Residual Momentum (Blitz-Huij-Martens 2011)
# ===================================================================

class ResidualMomentum(FactorBase):
    """12-1 month momentum **after stripping market and size betas**.

    References
    ----------
    Blitz, Huij & Martens (2011) "Residual Momentum", Journal of Empirical
    Finance.  The argument: vanilla price momentum is contaminated by
    cross-sectional dispersion in market and size factor exposures, which
    inflates volatility and produces well-known momentum crashes (Daniel &
    Moskowitz 2016).  Residualising on rolling Fama-French style factor
    returns isolates the *idiosyncratic* trend component, which has a
    higher Sharpe and a much smaller crash risk.

    Implementation
    --------------
    For each ticker on each date *t*:

    1. Build proxy market return = cross-sectional mean of daily returns.
    2. Build proxy size return = top-mcap-quartile minus bottom-quartile
       (only when ``market_cap`` is present in the price panel; otherwise
       only market is used).
    3. Regress trailing 36-month daily returns on these factors,
       residualise, then take the sum of residuals from t-252 to t-21
       (skip last month, the classic Jegadeesh-Titman "12-1" window).

    The *result* is the cumulative residual return over the same 12-1
    window — a per-ticker series.

    Notes
    -----
    The implementation is panel-aware: the proxies are built once globally
    and broadcast to all tickers, so this factor scales linearly in
    O(n_tickers * n_dates) rather than running a regression per (ticker, date).
    """

    meta = FactorMeta(
        name="RESID_MOM_12_1M",
        category="momentum",
        signal_type="cross_sectional",
        description="12-1 month residual momentum (Blitz-Huij-Martens 2011)",
        lookback=756,           # 36 months of daily history needed for stable betas
        lag=21,                 # skip the most recent month
        direction=1,
        references=[
            "Blitz, Huij & Martens (2011) Residual Momentum",
            "Daniel & Moskowitz (2016) Momentum Crashes",
        ],
    )

    _DEFAULT_REGRESSION_WINDOW = 504  # ~24 months — long enough for stable beta, short enough to react
    _DEFAULT_MOMENTUM_WINDOW = 252    # 12-month
    _DEFAULT_SKIP = 21                # skip 1 month

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals=None,
        _feature_cache=None,
    ) -> pd.Series:
        beta_window = int(self._extra_params.get("beta_window", self._DEFAULT_REGRESSION_WINDOW))
        mom_window = int(self._extra_params.get("mom_window", self._DEFAULT_MOMENTUM_WINDOW))
        skip = int(self._extra_params.get("skip", self._DEFAULT_SKIP))

        df = prices[["date", "ticker", "adj_close"]].copy()
        if "market_cap" in prices.columns:
            df["market_cap"] = prices["market_cap"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        ret = (
            df.groupby("ticker")["adj_close"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
        )
        df["ret"] = ret

        # --- Build factor proxies ---
        ret_wide = (
            df.pivot_table(index="date", columns="ticker", values="ret", aggfunc="first")
            .sort_index()
        )
        market = ret_wide.mean(axis=1)
        market.name = "mkt"

        if "market_cap" in df.columns and df["market_cap"].notna().any():
            mcap_wide = (
                df.pivot_table(index="date", columns="ticker", values="market_cap", aggfunc="first")
                .reindex_like(ret_wide)
            )
            # Top vs bottom quartile size return per date
            ranks = mcap_wide.rank(axis=1, pct=True)
            big_mask = ranks >= 0.75
            small_mask = ranks <= 0.25
            big_ret = ret_wide.where(big_mask).mean(axis=1)
            small_ret = ret_wide.where(small_mask).mean(axis=1)
            smb = small_ret - big_ret
            smb.name = "smb"
            factors = pd.concat([market, smb], axis=1)
        else:
            factors = market.to_frame()

        factors = factors.fillna(0.0)

        # --- Per-ticker rolling regression and residual cumulative return ---
        residual_cum = pd.Series(np.nan, index=df.index, dtype=float)
        for ticker, grp in df.groupby("ticker", sort=False):
            r = grp["ret"].to_numpy(dtype=float)
            n = len(r)
            if n < beta_window + skip + 5:
                continue
            f = factors.reindex(grp["date"]).to_numpy(dtype=float)  # (n, k)
            f = np.nan_to_num(f, nan=0.0)
            # Augment with intercept
            X = np.column_stack([np.ones(n), f])

            resid = np.full(n, np.nan, dtype=float)
            # Rolling betas using a single OLS at each anchor — only at points
            # where the cumulative residual is requested (we need the *last*
            # mom_window - skip residuals, computed from a beta fit on the
            # preceding beta_window window).  Since stocks rarely span >5
            # years, we approximate with a single beta per anchor and recompute
            # every 21 trading days for efficiency.
            for end in range(beta_window, n, 21):
                start = end - beta_window
                Xw = X[start:end]
                yw = r[start:end]
                mask = np.isfinite(yw)
                if mask.sum() < beta_window // 2:
                    continue
                try:
                    beta, *_ = np.linalg.lstsq(Xw[mask], yw[mask], rcond=None)
                except np.linalg.LinAlgError:
                    continue
                # Apply beta to the next 21-day block
                next_end = min(end + 21, n)
                X_block = X[end:next_end]
                y_block = r[end:next_end]
                resid_block = y_block - X_block @ beta
                resid[end:next_end] = resid_block

            # Cumulative residual return over [t - mom_window, t - skip]
            resid_series = pd.Series(resid, index=grp.index)
            cumulative = resid_series.rolling(mom_window - skip, min_periods=mom_window // 2).sum()
            cumulative = cumulative.shift(skip)
            residual_cum.loc[grp.index] = cumulative.to_numpy()

        return residual_cum


# ===================================================================
# Overnight return (Lou-Polk-Skouras 2019)
# ===================================================================

class OvernightReturn1M(FactorBase):
    """1-month average of overnight returns: open(t) / close(t-1) - 1.

    References
    ----------
    Lou, Polk & Skouras (2019) "A Tug of War: Overnight versus Intraday
    Expected Returns", Journal of Financial Economics.  The result: the
    cross-sectional return premium of stocks that beat the market overnight
    persists; intraday returns mean-revert.  In the era of dominant retail
    flow concentrated near US open, this signal is a clean proxy of demand
    pressure and remains tradable.

    The signal is the trailing 21-day mean of daily overnight log returns.
    Direction is +1 (higher overnight return → higher next-period return).
    """

    meta = FactorMeta(
        name="OVERNIGHT_RET_21D",
        category="microstructure",
        signal_type="cross_sectional",
        description="21-day mean overnight return: open(t)/close(t-1) - 1 (Lou-Polk-Skouras 2019)",
        lookback=22,
        lag=1,
        direction=1,
        references=["Lou, Polk & Skouras (2019) Tug of War"],
    )

    _DEFAULT_WINDOW = 21

    def _compute(
        self,
        prices: pd.DataFrame,
        fundamentals=None,
        _feature_cache=None,
    ) -> pd.Series:
        window = int(self._extra_params.get("window", self._DEFAULT_WINDOW))
        if "open" not in prices.columns:
            return pd.Series(np.nan, index=prices.index, dtype=float)

        df = prices[["date", "ticker", "open", "close"]].copy()
        if "adj_factor" in prices.columns:
            df["adj_factor"] = prices["adj_factor"]
        else:
            df["adj_factor"] = 1.0
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["ticker", "date"])

        # Apply same adj factor to open & close so overnight return is split-clean
        df["adj_open"] = df["open"] * df["adj_factor"]
        df["adj_close_raw"] = df["close"] * df["adj_factor"]
        df["prev_adj_close"] = df.groupby("ticker")["adj_close_raw"].shift(1)
        df["overnight_ret"] = df["adj_open"] / df["prev_adj_close"] - 1.0
        df["overnight_ret"] = df["overnight_ret"].replace([np.inf, -np.inf], np.nan)

        avg = (
            df.groupby("ticker")["overnight_ret"]
            .transform(lambda s: s.rolling(window, min_periods=max(5, window // 2)).mean())
        )
        return avg


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
    ResidualMomentum,
    OvernightReturn1M,
]
