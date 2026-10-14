"""Fetch and persist daily OHLCV price data for US equities.

Data source: ``yfinance`` (Yahoo Finance).

Key design choices
------------------
* **Raw + adjustment factor** — Yahoo's historical OHLCV is already expressed
  on the latest split basis even when ``auto_adjust=False``.  We first undo
  that provider-side split normalization, then store reconstructed historical
  (as-traded) OHLCV alongside explicit split and total-return factors.
* **Point-in-time adj_factor** — When ``point_in_time`` is ``True`` in
  config, the adjustment factor for each row reflects only corporate actions
  that had already occurred by that date.  Reconstructing the observable raw
  price is not a predictive input and prevents a split from creating a false
  return.
* **Calendar alignment** — Output is inner-joined to the NYSE trading
  calendar; weekends / holidays are never present.
"""

from __future__ import annotations

import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from quant_platform.core.data.calendar import align_to_calendar, trading_dates
from quant_platform.core.data.schema import PRICES_SCHEMA, validate_dataframe
from quant_platform.core.utils.io import load_config, resolve_data_path, save_parquet


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
    adjustment_method: str = "split_only",
    point_in_time: bool = True,
) -> Optional[pd.DataFrame]:
    """Download daily data for one ticker via ``yfinance``.

    Returns a DataFrame with columns matching :data:`PRICES_SCHEMA`,
    or ``None`` if the download fails after retries.
    """
    for attempt in range(1, max_retries + 1):
        try:
            obj = yf.Ticker(ticker)
            provider_history = obj.history(
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
            )
            if provider_history.empty:
                logger.warning("{}: empty history", ticker)
                return None
            raw = _restore_unadjusted_yahoo_history(provider_history)

            df = pd.DataFrame()
            df["date"] = raw.index.date
            df["ticker"] = ticker
            df["open"] = raw["Open"].values
            df["high"] = raw["High"].values
            df["low"] = raw["Low"].values
            df["close"] = raw["Close"].values
            df["volume"] = raw["Volume"].astype("int64").values
            df["amount"] = (raw["Close"] * raw["Volume"]).values

            df["split_factor"] = _cumulative_split_factor(raw).to_numpy(
                dtype=float
            )
            adj_factor, adj_close = _compute_adjustment_series(
                raw,
                method=adjustment_method,
                point_in_time=point_in_time,
            )
            df["adj_factor"] = adj_factor.to_numpy(dtype=float)
            df["adj_close"] = adj_close.to_numpy(dtype=float)

            # --- Trading status ---
            df["is_tradable"] = df["volume"] > 0

            return df

        except Exception as exc:
            logger.warning("{}: attempt {}/{} failed – {}", ticker, attempt, max_retries, exc)
            time.sleep(2 * attempt)

    logger.error("{}: all {} attempts failed", ticker, max_retries)
    return None


def _split_steps(raw: pd.DataFrame) -> pd.Series:
    """Return positive split ratios, with non-events represented by one."""
    values = raw.get("Stock Splits", 0.0)
    if isinstance(values, pd.Series):
        splits = pd.to_numeric(values, errors="coerce").fillna(0.0)
    else:
        splits = pd.Series(0.0, index=raw.index)
    return splits.where(splits > 0.0, 1.0).astype(float)


def _future_split_factor(raw: pd.DataFrame) -> pd.Series:
    """Splits strictly after each row, used to restore as-traded units."""
    steps = _split_steps(raw)
    return (
        steps.shift(-1, fill_value=1.0)
        .iloc[::-1]
        .cumprod()
        .iloc[::-1]
    )


def _cumulative_split_factor(raw: pd.DataFrame) -> pd.Series:
    """Splits effective on or before each row (a causal forward basis)."""
    return _split_steps(raw).cumprod()


def _restore_unadjusted_yahoo_history(history: pd.DataFrame) -> pd.DataFrame:
    """Undo Yahoo's built-in split normalization of historical OHLCV.

    ``auto_adjust=False`` disables dividend/``Adj Close`` auto-adjustment, but
    Yahoo still reports historical OHLC and volume on the latest split basis.
    Prices and cash dividends before a later split are therefore multiplied by
    the future split ratio, while volume is divided by it.  ``Adj Close`` is
    deliberately left on the provider basis for the explicitly non-PIT
    total-return option.
    """
    restored = history.copy()
    future_splits = _future_split_factor(restored)
    for column in ("Open", "High", "Low", "Close"):
        if column in restored.columns:
            restored[column] = (
                pd.to_numeric(restored[column], errors="coerce")
                * future_splits
            )
    if "Dividends" in restored.columns:
        restored["Dividends"] = (
            pd.to_numeric(restored["Dividends"], errors="coerce")
            * future_splits
        )
    if "Volume" in restored.columns:
        volume = (
            pd.to_numeric(restored["Volume"], errors="coerce")
            / future_splits.replace(0.0, np.nan)
        )
        restored["Volume"] = volume.round().fillna(0).astype("int64")
    return restored


def _compute_adjustment_series(
    raw: pd.DataFrame,
    method: str = "split_only",
    point_in_time: bool = True,
) -> tuple[pd.Series, pd.Series]:
    """Build split/total-return prices under an explicit adjustment policy."""
    if method not in {"split_only", "total_return"}:
        raise ValueError("adjustment.method must be 'split_only' or 'total_return'")

    close = pd.to_numeric(raw["Close"], errors="coerce").astype(float)
    split_steps = _split_steps(raw)

    if point_in_time:
        # Forward-scaled price: the factor changes only when the action becomes
        # public/effective, so an incremental refresh never rewrites history.
        factor = split_steps.cumprod()
        if method == "total_return":
            dividends = (
                pd.to_numeric(raw.get("Dividends", 0.0), errors="coerce")
                .fillna(0.0)
                if isinstance(raw.get("Dividends", 0.0), pd.Series)
                else pd.Series(0.0, index=raw.index)
            )
            reinvest_step = 1.0 + dividends / close.replace(0.0, np.nan)
            factor = factor * reinvest_step.fillna(1.0).cumprod()
        return factor, close * factor

    if method == "total_return" and "Adj Close" in raw.columns:
        adj_close = pd.to_numeric(raw["Adj Close"], errors="coerce").astype(float)
        factor = adj_close / close.replace(0.0, np.nan)
        return factor.fillna(1.0), adj_close

    # Conventional backward split adjustment for explicitly non-PIT research.
    future_splits = _future_split_factor(raw)
    factor = 1.0 / future_splits
    return factor, close * factor


def _calibrate_incremental_adjustments(
    old: pd.DataFrame,
    new: pd.DataFrame,
) -> pd.DataFrame:
    """Put newly downloaded PIT factors on each ticker's existing scale."""
    calibrated = new.copy()
    for ticker, new_group in calibrated.groupby("ticker", sort=False):
        old_group = old[old["ticker"] == ticker]
        if old_group.empty:
            continue
        calibration_columns = ["adj_factor"]
        if (
            "split_factor" in old_group.columns
            and "split_factor" in new_group.columns
        ):
            calibration_columns.append("split_factor")
        overlap = old_group[["date", *calibration_columns]].merge(
            new_group[["date", *calibration_columns]],
            on="date",
            suffixes=("_old", "_new"),
        ).dropna()
        overlap = overlap[
            overlap["adj_factor_new"].abs() > 1e-15
        ].sort_values("date")
        if overlap.empty:
            logger.warning("{}: no overlap available to calibrate adj_factor", ticker)
            continue
        anchor = overlap.iloc[0]
        scale = float(anchor["adj_factor_old"] / anchor["adj_factor_new"])
        mask = calibrated["ticker"].eq(ticker)
        calibrated.loc[mask, "adj_factor"] *= scale
        calibrated.loc[mask, "adj_close"] *= scale
        if "split_factor" in calibration_columns:
            split_anchor = overlap.loc[
                overlap["split_factor_new"].abs() > 1e-15
            ].iloc[0]
            split_scale = float(
                split_anchor["split_factor_old"]
                / split_anchor["split_factor_new"]
            )
            calibrated.loc[mask, "split_factor"] *= split_scale
    return calibrated


# ---------------------------------------------------------------------------
# Incremental download helpers
# ---------------------------------------------------------------------------

def _get_last_dates(existing_path: Path) -> Dict[str, pd.Timestamp]:
    """Read existing prices.parquet and return last date per ticker.
    
    Returns
    -------
    Dict mapping ticker -> last_date. Empty dict if file doesn't exist.
    """
    if not existing_path.exists():
        return {}
    
    try:
        from quant_platform.core.utils.io import load_parquet
        df = load_parquet(existing_path, columns=["date", "ticker"])
        df["date"] = pd.to_datetime(df["date"])
        last_dates = df.groupby("ticker")["date"].max().to_dict()
        logger.info("Found existing data for {} tickers, date range: {} to {}",
                    len(last_dates),
                    min(last_dates.values()).date() if last_dates else None,
                    max(last_dates.values()).date() if last_dates else None)
        return last_dates
    except Exception as exc:
        logger.warning("Failed to read existing prices: {}", exc)
        return {}


def _compute_download_start(
    ticker: str,
    last_dates: Dict[str, pd.Timestamp],
    global_start: str,
    overlap_days: int = 10,
) -> str:
    """Compute incremental download start date with overlap buffer.
    
    Parameters
    ----------
    ticker : ticker symbol
    last_dates : dict of ticker -> last_date from existing data
    global_start : fallback start date if no existing data
    overlap_days : trading days to overlap (for split/correction coverage)
    
    Returns
    -------
    Start date string for download (YYYY-MM-DD)
    """
    if ticker not in last_dates:
        return global_start
    
    # Compute overlap: last_date - overlap_days trading days
    last_date = last_dates[ticker]
    cal_dates = trading_dates()
    try:
        idx = cal_dates.get_loc(last_date)
        start_idx = max(0, idx - overlap_days)
        start_date = cal_dates[start_idx]
        return start_date.strftime("%Y-%m-%d")
    except Exception:
        # If last_date not in calendar, fall back to last_date - overlap_days calendar days
        start_date = last_date - pd.Timedelta(days=overlap_days * 1.5)  # rough estimate
        return start_date.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Batch download
# ---------------------------------------------------------------------------

def fetch_prices(
    tickers: Optional[List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    batch_size: int = 50,
    sleep_between: float = 1.0,
    incremental: bool = True,
    overlap_days: int = 10,
    existing_path: Optional[Path] = None,
    parallel: bool = False,
    max_workers: int = 4,
) -> pd.DataFrame:
    """Download OHLCV for *tickers* and return a single DataFrame.

    Parameters
    ----------
    tickers : list of ticker symbols.  Defaults to the index specified in
        ``config/settings.yaml``.
    start, end : date strings; defaults to config range.
    batch_size : how many tickers to process before sleeping (rate-limit).
        Only used if parallel=False.
    sleep_between : seconds to sleep between batches (only if parallel=False).
    incremental : if True, read existing data and only download from last_date.
    overlap_days : trading days to overlap for split/correction coverage.
    existing_path : path to existing prices.parquet for incremental mode.
    parallel : if True, use ThreadPoolExecutor for concurrent downloads.
    max_workers : number of threads for parallel downloads (default 4).
    """
    cfg = load_config()
    global_start = start or cfg["date_range"]["start"]
    end = end or cfg["date_range"]["end"]
    adjustment_method = cfg.get("adjustment", {}).get("method", "split_only")
    point_in_time = bool(cfg.get("adjustment", {}).get("point_in_time", True))
    if tickers is None:
        tickers = get_ticker_list(cfg["universe"]["index_membership"])

    # --- Incremental mode: get last dates ---
    last_dates: Dict[str, pd.Timestamp] = {}
    if incremental:
        if existing_path is None:
            existing_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
        last_dates = _get_last_dates(existing_path)
        if last_dates:
            import pyarrow.parquet as pq

            existing_columns = set(
                pq.ParquetFile(existing_path).schema.names
            )
            if "split_factor" not in existing_columns:
                raise ValueError(
                    "Existing price parquet predates the raw-price/split-factor "
                    "contract. Rebuild it with incremental=False before "
                    "appending new observations."
                )
            logger.info("Incremental mode: {} tickers have existing data, using overlap_days={}",
                        len(last_dates), overlap_days)

    # --- Download (serial or parallel) ---
    frames: list[pd.DataFrame] = []
    
    if parallel:
        logger.info("Parallel download with {} workers", max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for tkr in tickers:
                tkr_start = _compute_download_start(tkr, last_dates, global_start, overlap_days)
                future = executor.submit(
                    _download_single,
                    tkr,
                    tkr_start,
                    end,
                    3,
                    adjustment_method,
                    point_in_time,
                )
                futures[future] = tkr
            
            for future in tqdm(as_completed(futures), total=len(tickers), desc="Downloading prices"):
                result = future.result()
                if result is not None:
                    frames.append(result)
    else:
        # Serial download with rate limiting
        for i, tkr in enumerate(tqdm(tickers, desc="Downloading prices")):
            tkr_start = _compute_download_start(tkr, last_dates, global_start, overlap_days)
            result = _download_single(
                tkr,
                tkr_start,
                end,
                adjustment_method=adjustment_method,
                point_in_time=point_in_time,
            )
            if result is not None:
                frames.append(result)
            # Rate-limit
            if (i + 1) % batch_size == 0:
                time.sleep(sleep_between)

    if not frames:
        raise RuntimeError("No price data downloaded – check network / tickers.")

    df_new = pd.concat(frames, ignore_index=True)
    df_new["date"] = pd.to_datetime(df_new["date"])
    
    # --- Merge with existing data if incremental ---
    if incremental and last_dates and existing_path and existing_path.exists():
        from quant_platform.core.utils.io import load_parquet
        df_old = load_parquet(existing_path)
        df_old["date"] = pd.to_datetime(df_old["date"])
        if point_in_time:
            df_new = _calibrate_incremental_adjustments(df_old, df_new)
        
        # Concatenate and dedupe (keep newest download, not highest price)
        # df_new comes last in concat, so keep="last" preserves new downloads
        df = pd.concat([df_old, df_new], ignore_index=True)
        df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
        logger.info("Merged: {} old rows + {} new rows → {} total (after dedupe)",
                    len(df_old), len(df_new), len(df))
    else:
        df = df_new

    # Align to trading calendar
    df = align_to_calendar(
        df,
        date_col="date",
        start=global_start,
        end=end,
    )

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
    cfg = load_config()
    if output_path is None:
        output_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    saved_path = save_parquet(df, output_path, schema=PRICES_SCHEMA)
    adjustment = cfg.get("adjustment", {})
    metadata = {
        "data_source": "yfinance",
        "raw_ohlcv_adjusted": False,
        "provider_price_basis": "latest_split_adjusted",
        "stored_price_basis": "historical_as_traded_reconstructed",
        "split_factor_column": "split_factor",
        "adjustment_method": adjustment.get("method", "split_only"),
        "adjustment_point_in_time": bool(
            adjustment.get("point_in_time", True)
        ),
        "start_date": str(df["date"].min().date()),
        "end_date": str(df["date"].max().date()),
        "rows": int(len(df)),
        "tickers": int(df["ticker"].nunique()),
    }
    saved_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    return saved_path
