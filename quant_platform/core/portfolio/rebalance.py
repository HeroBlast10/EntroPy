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
        requested_end = (
            pd.Timestamp(end)
            if end is not None
            else pd.Timestamp(all_dates[-1])
        )
        # Extend beyond the requested boundary so a truncated Monday/Tuesday
        # is not mistaken for that week's final trading day.
        schedule_dates = trading_dates(
            start,
            str((requested_end + pd.Timedelta(days=8)).date()),
        )
        # Last trading day of each ISO week
        df = pd.DataFrame({"date": schedule_dates})
        df["year_week"] = df["date"].dt.isocalendar().year.astype(str) + "-" + \
                          df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
        last_per_week = df.groupby("year_week")["date"].max()
        result = pd.DatetimeIndex(
            sorted(last_per_week[last_per_week <= requested_end].values),
            name="date",
        )

    elif freq == "M":
        requested_end = (
            pd.Timestamp(end)
            if end is not None
            else pd.Timestamp(all_dates[-1])
        )
        schedule_dates = trading_dates(
            start,
            str((requested_end + pd.Timedelta(days=40)).date()),
        )
        # Last trading day of each calendar month
        df = pd.DataFrame({"date": schedule_dates})
        df["year_month"] = df["date"].dt.to_period("M")
        last_per_month = df.groupby("year_month")["date"].max()
        result = pd.DatetimeIndex(
            sorted(last_per_month[last_per_month <= requested_end].values),
            name="date",
        )

    else:
        raise ValueError(f"Unknown rebalance frequency: {freq!r}. Use 'D', 'W', or 'M'.")

    if len(result):
        logger.info(
            "Rebalance schedule: freq={}, {} dates ({} – {})",
            freq,
            len(result),
            result[0].date(),
            result[-1].date(),
        )
    else:
        logger.info("Rebalance schedule: freq={}, 0 complete periods", freq)
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

    # Melt back to long.  Preserve the rebalance schedule explicitly so the
    # execution engine can distinguish a held target from an instruction to
    # trade back to the same target after weights have drifted.
    long = wide.reset_index().melt(
        id_vars=["date"], var_name="ticker", value_name="weight",
    )
    rebalance_set = set(pd.to_datetime(rebalance_dates_list))
    long["is_rebalance"] = long["date"].isin(rebalance_set)

    # Drop inactive zero rows during invested periods.  When the whole target
    # is cash, retain the zero vector so the execution ledger still receives
    # the liquidation instruction and continues emitting cash-only dates.
    long = long.dropna(subset=["weight"])
    gross_by_date = long.groupby("date")["weight"].transform(
        lambda values: values.abs().sum()
    )
    long = long[
        (long["weight"].abs() > 1e-10)
        | (gross_by_date <= 1e-10)
    ].reset_index(drop=True)

    return long


def validate_portfolio_weights(
    weights: pd.DataFrame,
    mode: str = "long_only",
    tolerance: float = 1e-6,
    allow_cash: bool = False,
    config=None,
    sector_map: Optional[pd.DataFrame] = None,
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
                    "Long-only weights sum to {:.4f} (not 1.0) on {}. Tickers: {}",
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
            long_exposure = group.loc[group["weight"] > 0, "weight"].sum()
            short_exposure = group.loc[group["weight"] < 0, "weight"].sum()
            net_exposure = weight_sum
            gross_exposure = group["weight"].abs().sum()
            if gross_exposure > 2.0 + tolerance:
                raise ValueError(
                    f"Gross exposure {gross_exposure:.4f} > 2.0 on {date}"
                )
            if abs(net_exposure) > tolerance:
                raise ValueError(
                    f"Long-short net exposure {net_exposure:.4f} is not zero on {date}"
                )
            if allow_cash:
                if long_exposure > 1.0 + tolerance or short_exposure < -1.0 - tolerance:
                    raise ValueError(
                        f"Long-short sleeve exposure exceeds 100% on {date}"
                    )
            elif (
                abs(long_exposure - 1.0) > tolerance
                or abs(short_exposure + 1.0) > tolerance
            ):
                raise ValueError(
                    f"Long-short sleeves are not +1/-1 on {date}: "
                    f"long={long_exposure:.4f}, short={short_exposure:.4f}"
                )

        else:
            raise ValueError(f"Unknown portfolio mode: {mode!r}")

        if config is not None:
            if (group["weight"].abs() > config.max_stock_weight + tolerance).any():
                raise ValueError(f"max_stock_weight violated on {date}")

            if config.max_sector_weight is not None and config.max_sector_weight > 0:
                if sector_map is None or sector_map.empty:
                    raise ValueError(
                        "max_sector_weight is enabled but no sector_map was supplied"
                    )
                sector = (
                    sector_map[["ticker", "sector"]]
                    .drop_duplicates("ticker")
                    .set_index("ticker")["sector"]
                )
                mapped = group.set_index("ticker")[["weight"]].join(sector)
                if mapped["sector"].isna().any():
                    raise ValueError(
                        f"Missing sector mapping on {date}"
                    )
                gross = (
                    mapped.groupby("sector")["weight"]
                    .apply(lambda x: x.abs().sum())
                )
                if (gross > config.max_sector_weight + tolerance).any():
                    raise ValueError(f"max_sector_weight violated on {date}")
    
    logger.info(
        "Portfolio weights validated: mode={}, {} dates, all invariants satisfied",
        mode,
        weights["date"].nunique(),
    )
