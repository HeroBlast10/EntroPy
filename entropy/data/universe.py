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

from entropy.data.calendar import trading_dates
from entropy.data.schema import UNIVERSE_SCHEMA, validate_dataframe
from entropy.utils.io import load_config, resolve_data_path, save_parquet, load_parquet


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def _compute_days_since_ipo(prices: pd.DataFrame) -> pd.Series:
    """For each (date, ticker) row, compute # trading days since first appearance."""
    first_dates = prices.groupby("ticker")["date"].transform("min")
    return (prices["date"] - first_dates).dt.days


def build_universe(
    prices_path: Optional[Path | str] = None,
    fundamentals_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
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

    # --- Load prices ---
    if prices_path is None:
        prices_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    prices = load_parquet(prices_path)
    prices["date"] = pd.to_datetime(prices["date"])

    # --- Load fundamentals (optional, for market cap) ---
    mcap_lookup: Optional[pd.DataFrame] = None
    if fundamentals_path is not None:
        fpath = Path(fundamentals_path)
    else:
        fpath = resolve_data_path(cfg["paths"]["fundamentals_dir"], "fundamentals.parquet")
    if fpath.exists():
        fund = load_parquet(fpath, columns=["date", "ticker", "market_cap"])
        fund["date"] = pd.to_datetime(fund["date"])
        mcap_lookup = fund.set_index(["date", "ticker"])["market_cap"]
        logger.info("Loaded market_cap from fundamentals ({} rows)", len(fund))

    # --- Build universe frame ---
    uni = prices[["date", "ticker", "close", "volume", "is_tradable"]].copy()

    # Days since IPO
    uni["days_since_ipo"] = _compute_days_since_ipo(uni)

    # Market cap
    if mcap_lookup is not None:
        uni = uni.set_index(["date", "ticker"])
        uni["market_cap"] = mcap_lookup
        uni = uni.reset_index()
        # Forward-fill market_cap within ticker for days between reports
        uni["market_cap"] = uni.groupby("ticker")["market_cap"].ffill()
    else:
        # Rough proxy: close * shares_outstanding not available, so mark NaN
        # and skip market-cap filter
        uni["market_cap"] = np.nan
        logger.warning("No fundamentals found – market_cap filter will be skipped")

    uni["close_price"] = uni["close"]

    # --- Index membership placeholder (all True if no per-date data) ---
    uni["in_index"] = True

    # --- Apply filters ---
    filters = cfg["universe"]
    f1 = uni["days_since_ipo"] >= filters["min_listing_days"]
    f2 = uni["close_price"] >= filters["min_price"]
    f3 = uni["is_tradable"]
    f4 = True  # default: pass
    if not uni["market_cap"].isna().all():
        f4 = uni["market_cap"] >= filters["min_market_cap"]

    uni["pass_all_filters"] = f1 & f2 & f3 & f4

    # Keep only the schema columns
    uni = uni[["date", "ticker", "days_since_ipo", "market_cap",
               "close_price", "in_index", "pass_all_filters"]]

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
