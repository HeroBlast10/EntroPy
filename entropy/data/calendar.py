"""Trading calendar utilities built on ``exchange_calendars``.

Provides a thin wrapper that:
1. Returns trading dates as ``pd.DatetimeIndex`` for a given range.
2. Aligns an arbitrary DataFrame to the trading calendar (forward-fill or
   drop non-trading rows).
3. Answers point queries (is_trading_day, next/prev trading day).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import exchange_calendars as xcals
import pandas as pd
from loguru import logger

from entropy.utils.io import load_config


# ---------------------------------------------------------------------------
# Calendar singleton
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_calendar(name: Optional[str] = None) -> xcals.ExchangeCalendar:
    if name is None:
        cfg = load_config()
        name = cfg["exchange"]["calendar"]
    cal = xcals.get_calendar(name)
    logger.debug("Loaded exchange calendar: {}", name)
    return cal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def trading_dates(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> pd.DatetimeIndex:
    """Return a ``DatetimeIndex`` of trading days in [start, end].

    Defaults to the range specified in ``config/settings.yaml``.
    """
    cfg = load_config()
    start = pd.Timestamp(start or cfg["date_range"]["start"])
    end = pd.Timestamp(end or cfg["date_range"]["end"])
    cal = _get_calendar()
    sessions = cal.sessions_in_range(start, end)
    return pd.DatetimeIndex(sessions, name="date")


def is_trading_day(dt: str | pd.Timestamp) -> bool:
    cal = _get_calendar()
    ts = pd.Timestamp(dt)
    return cal.is_session(ts)


def prev_trading_day(dt: str | pd.Timestamp) -> pd.Timestamp:
    cal = _get_calendar()
    ts = pd.Timestamp(dt)
    return cal.previous_session(ts)


def next_trading_day(dt: str | pd.Timestamp) -> pd.Timestamp:
    cal = _get_calendar()
    ts = pd.Timestamp(dt)
    return cal.next_session(ts)


def align_to_calendar(
    df: pd.DataFrame,
    date_col: str = "date",
    method: str = "inner",
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Keep only rows whose *date_col* falls on a trading day.

    Parameters
    ----------
    df : DataFrame with a date column.
    date_col : name of the date column.
    method :
        ``"inner"`` – drop rows not on a trading day.
        ``"outer"`` – add rows for missing trading days (NaN-filled).
    start, end : override config date range.

    Returns
    -------
    DataFrame aligned to the trading calendar.
    """
    dates = trading_dates(start, end)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    if method == "inner":
        mask = df[date_col].isin(dates)
        dropped = (~mask).sum()
        if dropped:
            logger.debug("align_to_calendar: dropped {} non-trading-day rows", dropped)
        return df.loc[mask].reset_index(drop=True)

    elif method == "outer":
        # Build a full (date × ticker) grid if ticker exists
        if "ticker" in df.columns:
            tickers = df["ticker"].unique()
            idx = pd.MultiIndex.from_product(
                [dates, tickers], names=[date_col, "ticker"]
            )
            df = df.set_index([date_col, "ticker"]).reindex(idx).reset_index()
        else:
            df = df.set_index(date_col).reindex(dates).reset_index()
        return df

    raise ValueError(f"Unknown method: {method!r}")
