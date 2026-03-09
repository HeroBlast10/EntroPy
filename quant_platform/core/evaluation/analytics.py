"""Extended analytics: sector exposure, factor correlation, crowding proxy, rolling stats.

Complements the factor-level evaluation in ``quant_platform.core.signals.cross_sectional.evaluation``
with portfolio-level and cross-factor diagnostics that are commonly
asked about in quant interviews.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


# ===================================================================
# Sector / industry exposure
# ===================================================================

def sector_exposure(
    weights: pd.DataFrame,
    sector_map: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Compute aggregate portfolio weight per sector on each date.

    Parameters
    ----------
    weights : ``[date, ticker, weight]``
    sector_map : ``[ticker, sector]``

    Returns
    -------
    DataFrame ``[date, sector, weight_long, weight_short, weight_net]``
    """
    merged = weights.merge(sector_map, on=ticker_col, how="left")
    merged["sector"] = merged["sector"].fillna("Unknown")

    long = merged[merged["weight"] > 0].groupby([date_col, "sector"])["weight"].sum()
    short = merged[merged["weight"] < 0].groupby([date_col, "sector"])["weight"].sum()
    net = merged.groupby([date_col, "sector"])["weight"].sum()

    result = pd.DataFrame({
        "weight_long": long,
        "weight_short": short,
        "weight_net": net,
    }).fillna(0.0).reset_index()
    return result


def sector_exposure_summary(exposure: pd.DataFrame) -> pd.DataFrame:
    """Time-series average of sector exposure."""
    return exposure.groupby("sector")[["weight_long", "weight_short", "weight_net"]].mean()


# ===================================================================
# Factor correlation matrix
# ===================================================================

def factor_correlation(
    factor_df: pd.DataFrame,
    method: str = "spearman",
    date_col: str = "date",
) -> pd.DataFrame:
    """Compute average cross-sectional factor correlation.

    For each date, compute the correlation matrix across factors, then
    average over time.  Spearman (rank) correlation is more robust.
    """
    factor_cols = [c for c in factor_df.columns if c not in (date_col, "ticker")]

    corr_matrices = []
    for dt, grp in factor_df.groupby(date_col):
        sub = grp[factor_cols].dropna(how="all")
        if len(sub) >= 10:
            corr_matrices.append(sub.corr(method=method))

    if not corr_matrices:
        return pd.DataFrame()

    avg_corr = sum(corr_matrices) / len(corr_matrices)
    return avg_corr


# ===================================================================
# Rolling performance statistics
# ===================================================================

def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    annualisation: int = 252,
) -> pd.Series:
    """Rolling annualised Sharpe ratio."""
    roll_mean = returns.rolling(window, min_periods=window // 2).mean()
    roll_std = returns.rolling(window, min_periods=window // 2).std()
    return (roll_mean / roll_std) * np.sqrt(annualisation)


def rolling_volatility(
    returns: pd.Series,
    window: int = 60,
    annualisation: int = 252,
) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window, min_periods=window // 2).std() * np.sqrt(annualisation)


def rolling_turnover(
    weights: pd.DataFrame,
    window: int = 21,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.Series:
    """Rolling average one-way turnover over *window* rebalance periods."""
    dates = sorted(weights[date_col].unique())
    daily_to = {}
    prev = None
    for d in dates:
        cur = weights.loc[weights[date_col] == d].set_index(ticker_col)["weight"]
        if prev is not None:
            all_t = cur.index.union(prev.index)
            c = cur.reindex(all_t, fill_value=0.0)
            p = prev.reindex(all_t, fill_value=0.0)
            daily_to[d] = (c - p).abs().sum() / 2.0
        prev = cur
    ts = pd.Series(daily_to, name="turnover")
    ts.index = pd.DatetimeIndex(ts.index, name=date_col)
    return ts.rolling(window, min_periods=1).mean()


# ===================================================================
# Crowding proxy
# ===================================================================

def crowding_proxy(
    factor_df: pd.DataFrame,
    weights: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """Estimate factor crowding as the portfolio's exposure to each factor.

    Crowding = Σ(w_i × factor_zscore_i) for each factor on each date.
    A high value means the portfolio is heavily tilted toward that factor,
    which can indicate crowded positioning.

    Returns
    -------
    DataFrame ``[date, factor, exposure]``
    """
    factor_cols = [c for c in factor_df.columns if c not in (date_col, ticker_col)]
    merged = weights.merge(factor_df, on=[date_col, ticker_col], how="inner")

    records = []
    for dt, grp in merged.groupby(date_col):
        for fc in factor_cols:
            vals = grp[fc].dropna()
            w = grp.loc[vals.index, "weight"]
            exposure = (w * vals).sum()
            records.append({date_col: dt, "factor": fc, "exposure": exposure})

    return pd.DataFrame(records)


# ===================================================================
# Yearly / monthly return breakdown
# ===================================================================

def periodic_returns(
    daily_returns: pd.Series,
    freq: str = "Y",
) -> pd.Series:
    """Aggregate daily returns into yearly ('Y') or monthly ('M') returns."""
    dr = daily_returns.copy()
    dr.index = pd.DatetimeIndex(dr.index)

    if freq == "Y":
        grouped = dr.groupby(dr.index.year)
    elif freq == "M":
        grouped = dr.groupby(dr.index.to_period("M"))
    else:
        raise ValueError(f"Unknown freq: {freq}")

    return grouped.apply(lambda s: (1 + s).prod() - 1)


def monthly_return_table(daily_returns: pd.Series) -> pd.DataFrame:
    """Pivot monthly returns into a Year × Month table (heatmap-ready)."""
    dr = daily_returns.copy()
    dr.index = pd.DatetimeIndex(dr.index)
    dr_df = dr.to_frame("ret")
    dr_df["year"] = dr_df.index.year
    dr_df["month"] = dr_df.index.month

    monthly = dr_df.groupby(["year", "month"])["ret"].apply(lambda s: (1 + s).prod() - 1)
    return monthly.unstack(level="month")
