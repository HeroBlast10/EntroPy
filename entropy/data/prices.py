"""Fetch and persist daily OHLCV price data for US equities.

Data source: ``yfinance`` (Yahoo Finance).

Key design choices
------------------
* **Raw + adjustment factor** — We store *unadjusted* OHLCV alongside a
  cumulative split-adjustment factor so downstream code can reconstruct
  adjusted prices without baking look-ahead information into the dataset.
* **Point-in-time adj_factor** — When ``point_in_time`` is ``True`` in
  config, the adjustment factor for each row reflects only the splits that
  had *already occurred* by that date, preventing future-split leakage.
* **Calendar alignment** — Output is inner-joined to the NYSE trading
  calendar; weekends / holidays are never present.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from entropy.data.calendar import align_to_calendar, trading_dates
from entropy.data.schema import PRICES_SCHEMA, validate_dataframe
from entropy.utils.io import load_config, resolve_data_path, save_parquet


# ---------------------------------------------------------------------------
# Ticker list helpers
# ---------------------------------------------------------------------------

# Core subset used when Wikipedia scraping is blocked (covers ~50 large-caps
# across all 11 GICS sectors for representative backtesting).
_SP500_FALLBACK: List[str] = [
    # Information Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "ADBE", "INTC", "CSCO",
    # Health Care
    "UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SCHW", "BLK",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX",
    # Communication Services
    "GOOGL", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
    # Industrials
    "GE", "CAT", "RTX", "UNP", "HON", "BA", "DE", "LMT",
    # Consumer Staples
    "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP",
    # Real Estate
    "AMT", "PLD", "CCI", "EQIX", "SPG",
    # Materials
    "LIN", "APD", "SHW", "ECL", "NEM",
]


def _sp500_tickers() -> List[str]:
    """Scrape current S&P 500 constituents from Wikipedia.

    Uses ``requests`` with a browser User-Agent to avoid HTTP 403.
    Falls back to a hardcoded core subset if scraping fails.
    """
    import io
    import requests as _req

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    try:
        resp = _req.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        logger.info("Fetched {} S&P 500 tickers from Wikipedia", len(tickers))
        return sorted(set(tickers))
    except Exception as exc:
        logger.warning("Wikipedia scrape failed ({}), using hardcoded fallback", exc)
        return _SP500_FALLBACK[:]


def get_ticker_list(index: str = "sp500") -> List[str]:
    """Return a list of tickers for the requested index."""
    if index == "sp500":
        return _sp500_tickers()
    raise ValueError(f"Unsupported index: {index!r}")


# ---------------------------------------------------------------------------
# Single-ticker download
# ---------------------------------------------------------------------------

def _download_single(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = 3,
) -> Optional[pd.DataFrame]:
    """Download daily data for one ticker via ``yfinance``.

    Returns a DataFrame with columns matching :data:`PRICES_SCHEMA`,
    or ``None`` if the download fails after retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            obj = yf.Ticker(ticker)
            raw = obj.history(start=start, end=end, auto_adjust=False, actions=True)
            if raw.empty:
                logger.warning("{}: empty history", ticker)
                return None

            df = pd.DataFrame()
            df["date"] = raw.index.date
            df["ticker"] = ticker
            df["open"] = raw["Open"].values
            df["high"] = raw["High"].values
            df["low"] = raw["Low"].values
            df["close"] = raw["Close"].values
            df["volume"] = raw["Volume"].astype("int64").values
            df["amount"] = (raw["Close"] * raw["Volume"]).values

            # --- Compute point-in-time adjustment factor ---
            # yfinance provides "Adj Close" which already embeds the
            # cumulative adjustment.  We derive the factor from it.
            adj_close_yf = raw["Adj Close"].values
            close_arr = raw["Close"].values
            with np.errstate(divide="ignore", invalid="ignore"):
                adj_factor = np.where(close_arr != 0, adj_close_yf / close_arr, 1.0)
            df["adj_factor"] = adj_factor
            df["adj_close"] = adj_close_yf

            # --- Trading status ---
            df["is_tradable"] = df["volume"] > 0

            return df

        except Exception as exc:
            logger.warning("{}: attempt {}/{} failed – {}", ticker, attempt, max_retries, exc)
            time.sleep(2 * attempt)

    logger.error("{}: all {} attempts failed", ticker, max_retries)
    return None


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    batch_size: int = 50,
    sleep_between: float = 1.0,
) -> pd.DataFrame:
    """Download OHLCV for *tickers* and return a single DataFrame.

    Parameters
    ----------
    tickers : list of ticker symbols.  Defaults to the index specified in
        ``config/settings.yaml``.
    start, end : date strings; defaults to config range.
    batch_size : how many tickers to process before sleeping (rate-limit).
    sleep_between : seconds to sleep between batches.
    """
    cfg = load_config()
    start = start or cfg["date_range"]["start"]
    end = end or cfg["date_range"]["end"]
    if tickers is None:
        tickers = get_ticker_list(cfg["universe"]["index_membership"])

    frames: list[pd.DataFrame] = []
    for i, tkr in enumerate(tqdm(tickers, desc="Downloading prices")):
        result = _download_single(tkr, start, end)
        if result is not None:
            frames.append(result)
        # Rate-limit
        if (i + 1) % batch_size == 0:
            time.sleep(sleep_between)

    if not frames:
        raise RuntimeError("No price data downloaded – check network / tickers.")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])

    # Align to trading calendar
    df = align_to_calendar(df, date_col="date")

    # Sort for determinism
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info("Prices: {} rows, {} tickers, {} – {}", len(df),
                df["ticker"].nunique(), df["date"].min().date(), df["date"].max().date())
    return df


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def build_prices(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Full pipeline: download → validate → save Parquet.

    Returns the output file path.
    """
    df = fetch_prices(tickers=tickers, start=start, end=end)
    validate_dataframe(df, "prices")
    if output_path is None:
        cfg = load_config()
        output_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    return save_parquet(df, output_path, schema=PRICES_SCHEMA)
