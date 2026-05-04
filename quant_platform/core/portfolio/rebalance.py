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

import numpy as np

from quant_platform.core.data.calendar import trading_dates


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
    
    CRITICAL FIX: On each rebalance date, we explicitly zero out all stocks
    that are NOT in the new portfolio. This prevents old positions from being
    carried forward indefinitely.

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

    # Get all unique tickers that ever appear in the portfolio
    all_tickers = weights["ticker"].unique()
    
    # Get rebalance dates
    rebalance_dates_list = sorted(weights["date"].unique())
    
    # For each rebalance date, create a complete weight vector
    # (explicitly setting non-selected stocks to 0)
    complete_weights = []
    for reb_date in rebalance_dates_list:
        # Get weights for this rebalance date
        reb_weights = weights[weights["date"] == reb_date].set_index("ticker")["weight"]
        
        # Create complete vector: all tickers, with 0 for non-selected
        complete_vector = pd.Series(0.0, index=all_tickers)
        complete_vector.update(reb_weights)
        
        # Store as DataFrame rows
        for ticker in all_tickers:
            complete_weights.append({
                "date": reb_date,
                "ticker": ticker,
                "weight": complete_vector[ticker],
            })
    
    complete_df = pd.DataFrame(complete_weights)
    
    # Pivot to wide: date × ticker
    wide = complete_df.pivot(index="date", columns="ticker", values="weight")

    # Reindex to full calendar and forward-fill
    wide = wide.reindex(all_trading_dates).ffill()

    # Drop dates before the first rebalance
    first_reb = weights["date"].min()
    wide = wide.loc[wide.index >= first_reb]

    # Melt back to long
    long = wide.reset_index().melt(
        id_vars=["date"], var_name="ticker", value_name="weight",
    )

    # Drop zeros / NaN (stocks not in portfolio)
    # Keep only non-zero positions
    long = long.dropna(subset=["weight"])
    long = long[long["weight"].abs() > 1e-10].reset_index(drop=True)

    return long


def validate_portfolio_weights(
    weights: pd.DataFrame,
    mode: str = "long_only",
    tolerance: float = 1e-6,
    allow_cash: bool = False,
) -> None:
    """Validate portfolio weights satisfy invariants.
    
    Checks:
    - Long-only: sum(weights) = 1 on each date
    - Long-short: sum(abs(weights)) <= 2 (max gross exposure)
    - No negative weights in long-only mode
    
    Parameters
    ----------
    weights : DataFrame [date, ticker, weight]
    mode : "long_only" or "long_short"
    tolerance : numerical tolerance for sum checks
    
    Raises
    ------
    ValueError if invariants are violated
    """
    if weights.empty:
        return
    
    # Group by date and check invariants
    for date, group in weights.groupby("date"):
        weight_sum = group["weight"].sum()
        
        if mode == "long_only":
            # Check sum = 1 by default.  Regime overlays may intentionally
            # hold cash, in which case long-only gross exposure may be <= 1.
            invalid_sum = (
                weight_sum < -tolerance or weight_sum > 1.0 + tolerance
                if allow_cash
                else abs(weight_sum - 1.0) > tolerance
            )
            if invalid_sum:
                logger.error(
                    "Long-only weights sum to %.4f (not 1.0) on %s. Tickers: %s",
                    weight_sum,
                    date,
                    group[["ticker", "weight"]].to_dict("records"),
                )
                raise ValueError(
                    f"Long-only weights sum to {weight_sum:.4f} on {date}"
                )
            
            # Check no negative weights
            if (group["weight"] < -tolerance).any():
                neg = group[group["weight"] < -tolerance]
                raise ValueError(
                    f"Negative weights in long-only mode on {date}: {neg.to_dict('records')}"
                )
        
        elif mode == "long_short":
            # Check gross exposure <= 2 (100% long + 100% short)
            gross_exposure = group["weight"].abs().sum()
            if gross_exposure > 2.0 + tolerance:
                raise ValueError(
                    f"Gross exposure {gross_exposure:.4f} > 2.0 on {date}"
                )
    
    logger.info(
        "Portfolio weights validated: mode=%s, %d dates, all invariants satisfied",
        mode,
        weights["date"].nunique(),
    )
