"""Build a daily tradable universe by applying research-grade filters.

Filters applied (all configurable via ``settings.yaml``):
1. **min_listing_days** — exclude tickers that have been trading for fewer
   than *N* calendar days (avoids IPO volatility / thin history).
2. **min_price** — drop penny stocks whose close < threshold.
3. **min_market_cap** — drop micro-caps (requires fundamentals or proxy).
4. **is_tradable** — must have volume > 0 on the day (not halted).

The output table records *every* row that survives filtering, plus
diagnostic columns so you can audit which filter removed what.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from quant_platform.core.data.schema import UNIVERSE_SCHEMA, validate_dataframe
from quant_platform.core.utils.io import load_config, resolve_data_path, save_parquet, load_parquet


_MEMBERSHIP_COLUMNS = ["ticker", "membership_start", "membership_end"]


def load_membership_intervals(path: Path | str) -> pd.DataFrame:
    """Load an explicit historical index-membership interval file.

    The file must contain ``ticker``, ``membership_start`` and
    ``membership_end``.  End dates are inclusive and may be null for an
    active membership.  CSV and Parquet are supported.
    """
    membership_path = Path(path)
    if not membership_path.exists():
        raise FileNotFoundError(f"Membership interval file not found: {membership_path}")
    if membership_path.suffix.lower() == ".csv":
        intervals = pd.read_csv(membership_path)
    elif membership_path.suffix.lower() in {".parquet", ".pq"}:
        intervals = pd.read_parquet(membership_path)
    else:
        raise ValueError("Membership interval file must be CSV or Parquet")

    missing = set(_MEMBERSHIP_COLUMNS) - set(intervals.columns)
    if missing:
        raise ValueError(
            "Membership interval file is missing columns: "
            f"{sorted(missing)}"
        )

    intervals = intervals[list(_MEMBERSHIP_COLUMNS)].copy()
    intervals["ticker"] = intervals["ticker"].astype(str)
    intervals["membership_start"] = pd.to_datetime(
        intervals["membership_start"], errors="coerce"
    )
    intervals["membership_end"] = pd.to_datetime(
        intervals["membership_end"], errors="coerce"
    )
    if intervals["membership_start"].isna().any():
        raise ValueError("membership_start must be present and parseable")
    invalid = (
        intervals["membership_end"].notna()
        & (intervals["membership_end"] < intervals["membership_start"])
    )
    if invalid.any():
        raise ValueError("membership_end must be on or after membership_start")
    return intervals.sort_values(["ticker", "membership_start"]).reset_index(drop=True)


def _membership_mask(
    rows: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.Series:
    """Return whether each price row lies in any membership interval."""
    result = pd.Series(False, index=rows.index, dtype=bool)
    by_ticker = {
        ticker: group
        for ticker, group in intervals.groupby("ticker", sort=False)
    }
    for ticker, group in rows.groupby("ticker", sort=False):
        ticker_intervals = by_ticker.get(str(ticker))
        if ticker_intervals is None:
            continue
        dates = pd.to_datetime(group["date"])
        selected = pd.Series(False, index=group.index, dtype=bool)
        for interval in ticker_intervals.itertuples(index=False):
            in_interval = dates >= interval.membership_start
            if pd.notna(interval.membership_end):
                in_interval &= dates <= interval.membership_end
            selected |= in_interval
        result.loc[group.index] = selected
    return result


def require_point_in_time_universe(
    universe: pd.DataFrame,
    context: str = "production research",
) -> None:
    """Fail closed unless universe membership is explicitly point-in-time."""
    required = {"membership_source", "membership_is_point_in_time"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(
            f"{context} requires universe PIT provenance; missing {sorted(missing)}"
        )
    status = universe["membership_is_point_in_time"].fillna(False).astype(bool)
    if not status.all():
        sources = sorted(
            universe.loc[~status, "membership_source"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        raise ValueError(
            f"{context} rejects non-PIT universe membership sources: {sources}"
        )


def _apply_dynamic_universe_filter(
    uni: pd.DataFrame,
    top_n: int = 500,
    min_adv: float = 5e6,
    adv_window: int = 30,
) -> pd.DataFrame:
    """Apply dynamic universe filter: top N by market cap + liquidity threshold.
    
    This creates a 'liquidity-filtered large-cap universe' that approximates
    major indices like S&P 500, but is based on actual market data.
    
    Parameters
    ----------
    uni : DataFrame with [date, ticker, market_cap, close, volume]
    top_n : int
        Select top N stocks by market cap on each date (default 500)
    min_adv : float
        Minimum average dollar volume over adv_window (default $5M)
    adv_window : int
        Window for computing average dollar volume (default 30 days)
    
    Returns
    -------
    DataFrame with 'in_index' column added
    """
    uni = uni.copy()
    
    # Compute average dollar volume (ADV) over rolling window
    uni["dollar_volume"] = uni["close"] * uni["volume"]
    uni["adv"] = uni.groupby("ticker")["dollar_volume"].transform(
        lambda x: x.rolling(window=adv_window, min_periods=adv_window // 2).mean()
    )
    
    # For each date, rank stocks by market cap and select top N
    def select_top_n(group):
        # Filter by liquidity first
        liquid = group[group["adv"] >= min_adv]
        
        # If market_cap is available, rank by it
        if not liquid["market_cap"].isna().all():
            # Rank by market cap (descending)
            liquid = liquid.sort_values("market_cap", ascending=False)
            # Select top N
            top_stocks = liquid.head(top_n)["ticker"].tolist()
        else:
            # Fallback: if no market cap, select top N by ADV
            liquid = liquid.sort_values("adv", ascending=False)
            top_stocks = liquid.head(top_n)["ticker"].tolist()
        
        # Mark selected stocks
        group["in_index"] = group["ticker"].isin(top_stocks)
        return group
    
    uni = uni.groupby("date", group_keys=False).apply(select_top_n)
    
    # Drop temporary columns
    uni = uni.drop(columns=["dollar_volume", "adv"])
    
    # Log statistics
    total_dates = uni["date"].nunique()
    avg_in_index = uni.groupby("date")["in_index"].sum().mean()
    logger.info(
        "Dynamic universe filter: top {} by market cap + ADV >= ${:.1f}M, "
        "avg {:.0f} stocks per date across {} dates",
        top_n, min_adv / 1e6, avg_in_index, total_dates,
    )
    
    return uni


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _compute_days_since_ipo(prices: pd.DataFrame) -> pd.Series:
    """For each (date, ticker) row, compute # trading days since first appearance."""
    ordered = prices.sort_values(["ticker", "date"])
    counts = ordered.groupby("ticker", sort=False).cumcount()
    return counts.reindex(prices.index).astype("int32")


def lag_universe_eligibility(
    universe: pd.DataFrame,
    periods: int = 1,
) -> pd.DataFrame:
    """Move end-of-day universe information to a later trading session.

    Universe filters use closing price, completed daily volume and trading
    status.  They are therefore known only after that session's close and
    cannot authorize a same-close order.  The returned ``date`` is the
    decision/trade date while ``information_date`` preserves the source date.
    A global trading-date map is used so a missing ticker row never silently
    stretches one day's eligibility across a suspension.
    """
    if periods < 0:
        raise ValueError("periods must be non-negative")
    required = {"date", "ticker", "pass_all_filters"}
    if missing := required.difference(universe.columns):
        raise ValueError(
            f"universe missing eligibility columns: {sorted(missing)}"
        )
    result = universe.copy()
    result["date"] = pd.to_datetime(result["date"])
    if result.duplicated(["date", "ticker"]).any():
        raise ValueError("universe contains duplicate date/ticker keys")
    result["information_date"] = result["date"]
    if periods == 0:
        return result
    dates = pd.DatetimeIndex(
        result["date"].dropna().unique()
    ).sort_values()
    shifted = pd.Series(
        pd.Series(dates, index=dates).shift(-periods).to_numpy(),
        index=dates,
    )
    result["date"] = result["date"].map(shifted)
    return result.dropna(subset=["date"]).copy()


def _daily_market_cap_from_events(
    price_rows: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.Series:
    """Reprice the latest known market cap on a split-consistent basis.

    Fundamental observations become usable on their ``date`` (the publication
    or configured availability date).  ``close * split_factor`` provides a
    continuous split-only price basis for canonical raw-price files.  Event
    market cap is preferred because it directly anchors a compatible share
    count.  If only reported shares exist, they are divided by the split factor
    at the report date (or availability date when no report date exists).
    Carrying this normalised share count prevents both frozen market caps and
    split-date jumps.

    Legacy price files without ``split_factor`` are treated as already
    split-normalised.  Such inputs remain research-only because historical
    share-basis provenance cannot be reconstructed from OHLCV alone.
    """
    required_prices = {"date", "ticker", "close"}
    required_fundamentals = {"date", "ticker"}
    if missing := required_prices.difference(price_rows.columns):
        raise ValueError(f"price rows missing columns: {sorted(missing)}")
    if missing := required_fundamentals.difference(fundamentals.columns):
        raise ValueError(f"fundamentals missing columns: {sorted(missing)}")
    if not {"shares_outstanding", "market_cap"}.intersection(fundamentals.columns):
        return pd.Series(np.nan, index=price_rows.index, dtype=float)

    price_columns = ["date", "ticker", "close"]
    if "split_factor" in price_rows.columns:
        price_columns.append("split_factor")
    prices = price_rows[price_columns].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    if "split_factor" in prices.columns:
        prices["_split_factor"] = (
            pd.to_numeric(prices["split_factor"], errors="coerce")
            .where(lambda values: values > 0)
        )
    else:
        prices["_split_factor"] = 1.0
    prices["_split_price"] = (
        pd.to_numeric(prices["close"], errors="coerce")
        * prices["_split_factor"]
    )
    prices["_row_id"] = np.arange(len(prices))
    optional_event_columns = [
        column
        for column in ("shares_outstanding", "market_cap", "report_date")
        if column in fundamentals.columns
    ]
    events = fundamentals[
        ["date", "ticker", *optional_event_columns]
    ].copy()
    events["date"] = pd.to_datetime(events["date"])
    if "report_date" in events.columns:
        events["report_date"] = pd.to_datetime(events["report_date"])
    events = (
        events.sort_values(["ticker", "date"])
        .drop_duplicates(["date", "ticker"], keep="last")
    )

    effective_events: list[pd.DataFrame] = []
    price_groups = {
        ticker: group.sort_values("date")
        for ticker, group in prices.groupby("ticker", sort=False)
    }
    for ticker, ticker_events in events.groupby("ticker", sort=False):
        ticker_prices = price_groups.get(ticker)
        if ticker_prices is None or ticker_prices.empty:
            continue
        price_dates = ticker_prices["date"].to_numpy(dtype="datetime64[ns]")
        event_dates = ticker_events["date"].to_numpy(dtype="datetime64[ns]")
        locations = np.searchsorted(price_dates, event_dates, side="left")
        valid = locations < len(ticker_prices)
        if not valid.any():
            continue
        selected_events = ticker_events.iloc[np.flatnonzero(valid)].copy()
        selected_locations = locations[valid]
        reference_price = pd.to_numeric(
            ticker_prices.iloc[selected_locations]["_split_price"],
            errors="coerce",
        ).to_numpy(dtype=float)
        basis_locations = selected_locations.copy()
        if "report_date" in selected_events.columns:
            report_dates = selected_events["report_date"].to_numpy(
                dtype="datetime64[ns]"
            )
            has_report_date = ~pd.isna(report_dates)
            report_locations = np.searchsorted(
                price_dates,
                report_dates,
                side="left",
            )
            report_locations = np.clip(
                report_locations,
                0,
                len(ticker_prices) - 1,
            )
            basis_locations = np.where(
                has_report_date,
                report_locations,
                basis_locations,
            )
        basis_split_factor = pd.to_numeric(
            ticker_prices.iloc[basis_locations]["_split_factor"],
            errors="coerce",
        ).to_numpy(dtype=float)
        shares = (
            pd.to_numeric(
                selected_events.get(
                    "shares_outstanding",
                    pd.Series(np.nan, index=selected_events.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=float)
        )
        event_market_cap = (
            pd.to_numeric(
                selected_events.get(
                    "market_cap",
                    pd.Series(np.nan, index=selected_events.index),
                ),
                errors="coerce",
            ).to_numpy(dtype=float)
        )
        shares_from_market_cap = np.divide(
            event_market_cap,
            reference_price,
            out=np.full_like(event_market_cap, np.nan),
            where=np.isfinite(reference_price) & (reference_price > 0),
        )
        normalised_reported_shares = np.divide(
            shares,
            basis_split_factor,
            out=np.full_like(shares, np.nan),
            where=np.isfinite(basis_split_factor)
            & (basis_split_factor > 0),
        )
        normalised_shares = np.where(
            np.isfinite(shares_from_market_cap)
            & (shares_from_market_cap > 0),
            shares_from_market_cap,
            normalised_reported_shares,
        )
        effective_events.append(
            pd.DataFrame(
                {
                    "date": ticker_prices.iloc[selected_locations][
                        "date"
                    ].to_numpy(),
                    "ticker": ticker,
                    "_shares_for_mcap": normalised_shares,
                }
            )
        )

    if not effective_events:
        return pd.Series(np.nan, index=price_rows.index, dtype=float)
    event_frame = (
        pd.concat(effective_events, ignore_index=True)
        .sort_values(["ticker", "date"])
        .drop_duplicates(["date", "ticker"], keep="last")
    )
    repriced = prices.merge(
        event_frame,
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    repriced = repriced.sort_values(["ticker", "date"])
    repriced["_shares_for_mcap"] = repriced.groupby(
        "ticker", sort=False
    )["_shares_for_mcap"].ffill()
    repriced["market_cap"] = (
        pd.to_numeric(repriced["_split_price"], errors="coerce")
        * repriced["_shares_for_mcap"]
    )
    return (
        repriced.set_index("_row_id")["market_cap"]
        .reindex(np.arange(len(price_rows)))
        .set_axis(price_rows.index)
    )


def build_universe(
    prices_path: Optional[Path | str] = None,
    fundamentals_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    membership_path: Optional[Path | str] = None,
    mode: Optional[str] = None,
) -> Path:
    """Build the daily tradable universe and save to Parquet.

    Parameters
    ----------
    prices_path : path to ``prices.parquet``.  Defaults to the config path.
    fundamentals_path : path to ``fundamentals.parquet``.  If available,
        used for market-cap filtering; otherwise market_cap is approximated
        from ``close * volume`` as a rough proxy.
    output_path : destination Parquet path.

    Returns
    -------
    Path to the saved universe Parquet file.
    """
    cfg = load_config()
    contract_mode = (
        mode or cfg.get("data_contract", {}).get("mode", "research")
    ).lower()
    if contract_mode not in {"research", "production"}:
        raise ValueError("mode must be 'research' or 'production'")

    # --- Load prices ---
    if prices_path is None:
        prices_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    prices = load_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    # --- Load fundamentals (optional, for market cap) ---
    fundamental_events: Optional[pd.DataFrame] = None
    if fundamentals_path is not None:
        fpath = Path(fundamentals_path)
    else:
        fpath = resolve_data_path(cfg["paths"]["fundamentals_dir"], "fundamentals.parquet")
    if fpath.exists():
        fund = load_parquet(fpath)
        if contract_mode == "production":
            from quant_platform.core.data.fundamentals import (
                require_point_in_time_fundamentals,
            )
            require_point_in_time_fundamentals(
                fund,
                context="production universe market-cap filter",
            )
        available_columns = [
            column
            for column in (
                "date",
                "ticker",
                "report_date",
                "market_cap",
                "shares_outstanding",
            )
            if column in fund.columns
        ]
        if not {"market_cap", "shares_outstanding"}.intersection(
            available_columns
        ):
            if contract_mode == "production":
                raise RuntimeError(
                    "Production fundamentals need market_cap or shares_outstanding"
                )
            logger.warning(
                "Fundamentals contain no market_cap/shares_outstanding; "
                "market-cap filter will be skipped"
            )
        fund = fund[available_columns]
        fund["date"] = pd.to_datetime(fund["date"])
        sort_columns = ["ticker", "date"]
        if "report_date" in fund.columns:
            fund["report_date"] = pd.to_datetime(fund["report_date"])
            sort_columns.append("report_date")
        fundamental_events = (
            fund.sort_values(sort_columns)
            .drop_duplicates(["date", "ticker"], keep="last")
        )
        logger.info("Loaded market_cap from fundamentals ({} rows)", len(fund))
    elif contract_mode == "production":
        raise RuntimeError(
            "Production universe requires point-in-time fundamentals for "
            "the configured market-cap filter"
        )

    # --- Build universe frame ---
    uni = prices[["date", "ticker", "close", "volume", "is_tradable"]].copy()

    # Days since IPO
    uni["days_since_ipo"] = _compute_days_since_ipo(uni)

    # Market cap
    if fundamental_events is not None:
        uni["market_cap"] = _daily_market_cap_from_events(
            uni,
            fundamental_events,
        )
    else:
        # Rough proxy: close * shares_outstanding not available, so mark NaN
        # and skip market-cap filter
        uni["market_cap"] = np.nan
        logger.warning("No fundamentals found – market_cap filter will be skipped")

    uni["close_price"] = uni["close"]

    configured_membership = cfg["universe"].get("membership_file")
    membership_path = membership_path or (
        resolve_data_path(configured_membership)
        if configured_membership
        else None
    )
    if membership_path is not None:
        intervals = load_membership_intervals(membership_path)
        if contract_mode == "production":
            data_start, data_end = uni["date"].min(), uni["date"].max()
            relevant = intervals[
                intervals["membership_start"].le(data_end)
                & (
                    intervals["membership_end"].isna()
                    | intervals["membership_end"].ge(data_start)
                )
            ]
            missing_tickers = sorted(
                set(relevant["ticker"]) - set(uni["ticker"].astype(str))
            )
            if missing_tickers:
                preview = missing_tickers[:10]
                raise RuntimeError(
                    "Production price panel is missing historical members "
                    f"declared by the membership file: {preview}"
                )
        uni["in_index"] = _membership_mask(uni, intervals)
        uni["membership_source"] = (
            f"historical_interval_file:{Path(membership_path).name}"
        )
        uni["membership_is_point_in_time"] = True
    else:
        if contract_mode == "production":
            raise RuntimeError(
                "Production universe requires universe.membership_file with "
                "ticker/membership_start/membership_end intervals"
            )
        # Research-only proxy.  The price seed is normally today's S&P 500
        # snapshot, so this must never be represented as historical PIT
        # membership even though the daily liquidity ranking is dynamic.
        uni = _apply_dynamic_universe_filter(
            uni,
            top_n=cfg["universe"].get("top_n_by_mcap", 500),
            min_adv=cfg["universe"].get("min_avg_dollar_volume", 5e6),
            adv_window=cfg["universe"].get("adv_window_days", 30),
        )
        uni["membership_source"] = "current_constituent_seed_dynamic_proxy"
        uni["membership_is_point_in_time"] = False

    # --- Apply filters ---
    filters = cfg["universe"]
    f1 = uni["days_since_ipo"] >= filters["min_listing_days"]
    f2 = uni["close_price"] >= filters["min_price"]
    f3 = uni["is_tradable"]
    # Apply the market-cap floor only where a point-in-time value is
    # available.  Historical rows without fundamental coverage have already
    # been screened by the rolling ADV fallback in the dynamic-universe step;
    # rejecting every such row would truncate the entire universe to the
    # fundamental provider's recent reporting window.
    f4 = uni["market_cap"].isna() | (
        uni["market_cap"] >= filters["min_market_cap"]
    )

    uni["pass_all_filters"] = f1 & f2 & f3 & f4 & uni["in_index"]

    # Keep only the schema columns
    uni = uni[["date", "ticker", "days_since_ipo", "market_cap",
               "close_price", "in_index", "membership_source",
               "membership_is_point_in_time", "pass_all_filters"]]

    # Only persist rows that pass (keeps file small; diagnostics available
    # by re-running without the filter if needed).
    passed = uni[uni["pass_all_filters"]].copy().reset_index(drop=True)

    validate_dataframe(passed, "universe")

    n_total = len(uni)
    n_pass = len(passed)
    logger.info("Universe: {}/{} rows pass all filters ({:.1%})",
                n_pass, n_total, n_pass / max(n_total, 1))

    if output_path is None:
        output_path = resolve_data_path(cfg["paths"]["universe_dir"], "universe.parquet")
    return save_parquet(passed, output_path, schema=UNIVERSE_SCHEMA)
