"""Rebalance schedule generation.

Supports three frequencies:
- **D** (daily)   — rebalance every trading day
- **W** (weekly)  — rebalance on the last trading day of each week
- **M** (monthly) — rebalance on the last trading day of each month

All dates are guaranteed to fall on NYSE trading days.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from loguru import logger

from entropy.data.calendar import trading_dates


def rebalance_dates(
    freq: str = "M",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DatetimeIndex:
    """Generate a rebalance schedule aligned to the trading calendar.

    Parameters
    ----------
    freq : ``"D"`` | ``"W"`` | ``"M"``
    start, end : date strings; defaults to config range.

    Returns
    -------
    ``pd.DatetimeIndex`` of rebalance dates.
    """
    all_dates = trading_dates(start, end)

    if freq == "D":
        result = all_dates

    elif freq == "W":
        # Last trading day of each ISO week
        df = pd.DataFrame({"date": all_dates})
        df["year_week"] = df["date"].dt.isocalendar().year.astype(str) + "-" + \
                          df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        last_per_week = df.groupby("year_week")["date"].max()
        result = pd.DatetimeIndex(sorted(last_per_week.values), name="date")

    elif freq == "M":
        # Last trading day of each calendar month
        df = pd.DataFrame({"date": all_dates})
        df["year_month"] = df["date"].dt.to_period("M")
        last_per_month = df.groupby("year_month")["date"].max()
        result = pd.DatetimeIndex(sorted(last_per_month.values), name="date")

    else:
        raise ValueError(f"Unknown rebalance frequency: {freq!r}. Use 'D', 'W', or 'M'.")

    logger.info("Rebalance schedule: freq={}, {} dates ({} – {})",
                freq, len(result), result[0].date(), result[-1].date())
    return result


def carry_forward_weights(
    weights: pd.DataFrame,
    all_trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Expand rebalance-day weights to every trading day via forward-fill.

    Between rebalances the portfolio holds steady (weights unchanged).

    Parameters
    ----------
    weights : ``[date, ticker, weight]`` on rebalance dates only.
    all_trading_dates : full set of trading dates.

    Returns
    -------
    ``[date, ticker, weight]`` on every trading day.
    """
    if weights.empty:
        return weights

    weights = weights.copy()
    weights["date"] = pd.to_datetime(weights["date"])

    # Pivot to wide: date × ticker
    wide = weights.pivot(index="date", columns="ticker", values="weight")

    # Reindex to full calendar and forward-fill
    wide = wide.reindex(all_trading_dates).ffill()

    # Drop dates before the first rebalance
    first_reb = weights["date"].min()
    wide = wide.loc[wide.index >= first_reb]

    # Melt back to long
    long = wide.reset_index().melt(
        id_vars=["index"], var_name="ticker", value_name="weight",
    ).rename(columns={"index": "date"})

    # Drop zeros / NaN (stocks not in portfolio)
    long = long.dropna(subset=["weight"])
    long = long[long["weight"].abs() > 1e-10].reset_index(drop=True)

    return long
