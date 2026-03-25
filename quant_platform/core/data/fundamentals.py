"""Fetch and persist point-in-time fundamental data for US equities.

Data source: **SimFin** (free tier).  Falls back to ``yfinance`` for
market-cap / shares-outstanding when SimFin data is unavailable.

Point-in-time handling
----------------------
Financial statements have a *report_date* (fiscal period end) and a
*publish_date* (the date the filing becomes public).  To avoid look-ahead
bias we only "see" a report starting on::

    available_date = publish_date + publication_lag_days

This ``available_date`` is mapped to the next trading day and stored in
the ``date`` column of the output table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from tqdm import tqdm

from quant_platform.core.data.calendar import align_to_calendar, next_trading_day, trading_dates
from quant_platform.core.data.schema import FUNDAMENTALS_SCHEMA, validate_dataframe
from quant_platform.core.utils.io import load_config, resolve_data_path, save_parquet

# ---------------------------------------------------------------------------
# SimFin helpers
# ---------------------------------------------------------------------------

_SIMFIN_AVAILABLE = False

try:
    import simfin as sf
    _SIMFIN_AVAILABLE = True
except ImportError:
    logger.warning("simfin not installed – fundamentals will use yfinance fallback only")


def _setup_simfin(api_key: Optional[str] = None, data_dir: Optional[str] = None) -> None:
    """Configure SimFin API key and cache directory."""
    if not _SIMFIN_AVAILABLE:
        return
    key = api_key or os.environ.get("SIMFIN_API_KEY", "free")
    sf.set_api_key(key)
    # SimFin requires a data directory; default to ~/simfin_data/
    if data_dir is None:
        data_dir = os.path.join(os.path.expanduser("~"), "simfin_data")
    sf.set_data_dir(data_dir)
    logger.debug("SimFin configured (api_key={}…, data_dir={})", key[:4], data_dir)


def _fetch_simfin_statements(variant: str = "free") -> Dict[str, pd.DataFrame]:
    """Download income / balance / cashflow from SimFin bulk datasets."""
    if not _SIMFIN_AVAILABLE:
        return {}

    market = "us"
    frames: Dict[str, pd.DataFrame] = {}

    try:
        frames["income"] = sf.load(
            dataset="income", variant=variant, market=market,
        )
    except Exception as exc:
        logger.warning("SimFin income load failed: {}", exc)

    try:
        frames["balance"] = sf.load(
            dataset="balance", variant=variant, market=market,
        )
    except Exception as exc:
        logger.warning("SimFin balance load failed: {}", exc)

    try:
        frames["cashflow"] = sf.load(
            dataset="cashflow", variant=variant, market=market,
        )
    except Exception as exc:
        logger.warning("SimFin cashflow load failed: {}", exc)

    return frames


# ---------------------------------------------------------------------------
# yfinance fallback for market cap / shares outstanding
# ---------------------------------------------------------------------------

def _fetch_market_cap_yf(
    tickers: List[str],
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Use yfinance to build a (date, ticker, market_cap, shares_outstanding) table.

    This is a rough fallback: yfinance only provides the *current* shares
    outstanding.  We approximate historical market cap as
    ``close_on_date × current_shares_outstanding``, which is imperfect but
    sufficient for universe filtering.
    """
    records = []
    for tkr in tqdm(tickers, desc="Fetching market cap (yfinance)"):
        try:
            info = yf.Ticker(tkr).info
            shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if shares is None:
                continue
            records.append({"ticker": tkr, "shares_outstanding": float(shares)})
        except Exception:
            continue

    if not records:
        return pd.DataFrame(columns=["date", "ticker", "market_cap", "shares_outstanding"])

    shares_df = pd.DataFrame(records)
    # Cross-join with dates (market_cap filled later via prices)
    shares_df["_key"] = 1
    date_df = pd.DataFrame({"date": dates, "_key": 1})
    out = date_df.merge(shares_df, on="_key").drop(columns="_key")
    out["market_cap"] = np.nan  # will be filled by caller using close prices
    return out


# ---------------------------------------------------------------------------
# yfinance fallback for financial statements
# ---------------------------------------------------------------------------

def _fetch_financials_yf(
    tickers: List[str],
    lag_days: int,
) -> pd.DataFrame:
    """Fetch quarterly financial statements from yfinance as a fallback.

    Pulls income statement, balance sheet, and cash flow for each ticker,
    then maps them to the project schema.  ``publish_date`` is approximated
    as ``report_date + 45 days`` (SEC 10-Q deadline for large accelerated
    filers), and point-in-time lag is applied on top.
    """
    all_records: List[Dict] = []

    for tkr in tqdm(tickers, desc="Fetching financials (yfinance)"):
        try:
            t = yf.Ticker(tkr)

            # --- Income statement (quarterly) ---
            inc = t.quarterly_income_stmt
            bal = t.quarterly_balance_sheet
            cf = t.quarterly_cashflow

            # Collect all report dates across statements
            report_dates = set()
            if inc is not None and not inc.empty:
                report_dates.update(inc.columns)
            if bal is not None and not bal.empty:
                report_dates.update(bal.columns)
            if cf is not None and not cf.empty:
                report_dates.update(cf.columns)

            for rd in report_dates:
                row: Dict = {
                    "ticker": tkr,
                    "report_date": pd.Timestamp(rd),
                }

                # Income statement fields
                if inc is not None and rd in inc.columns:
                    col = inc[rd]
                    row["revenue"] = _safe_get(col, ["Total Revenue", "Revenue"])
                    row["gross_profit"] = _safe_get(col, ["Gross Profit"])
                    row["operating_income"] = _safe_get(
                        col, ["Operating Income", "EBIT"]
                    )
                    row["net_income"] = _safe_get(
                        col, ["Net Income", "Net Income Common Stockholders"]
                    )
                    row["eps_diluted"] = _safe_get(
                        col, ["Diluted EPS", "Basic EPS"]
                    )

                # Balance sheet fields
                if bal is not None and rd in bal.columns:
                    col = bal[rd]
                    row["total_assets"] = _safe_get(col, ["Total Assets"])
                    row["total_liabilities"] = _safe_get(
                        col,
                        ["Total Liabilities Net Minority Interest",
                         "Total Liab"],
                    )
                    row["total_equity"] = _safe_get(
                        col,
                        ["Total Equity Gross Minority Interest",
                         "Stockholders Equity",
                         "Total Stockholder Equity"],
                    )
                    row["cash_and_equivalents"] = _safe_get(
                        col,
                        ["Cash And Cash Equivalents",
                         "Cash Cash Equivalents And Short Term Investments"],
                    )
                    row["total_debt"] = _safe_get(col, ["Total Debt"])
                    row["shares_outstanding"] = _safe_get(
                        col, ["Share Issued", "Ordinary Shares Number"]
                    )

                # Cash flow fields
                if cf is not None and rd in cf.columns:
                    col = cf[rd]
                    row["cash_from_operations"] = _safe_get(
                        col,
                        ["Operating Cash Flow",
                         "Cash Flowsfromusedin Operating Activities Direct"],
                    )
                    row["capex"] = _safe_get(
                        col, ["Capital Expenditure"]
                    )
                    row["free_cash_flow"] = _safe_get(
                        col, ["Free Cash Flow"]
                    )

                all_records.append(row)

        except Exception as exc:
            logger.debug("yfinance financials failed for {}: {}", tkr, exc)
            continue

    if not all_records:
        logger.warning("yfinance returned no financial statements")
        return pd.DataFrame()

    fund = pd.DataFrame(all_records)
    fund["report_date"] = pd.to_datetime(fund["report_date"])

    # Approximate publish_date as report_date + 45 days (SEC 10-Q deadline)
    fund["publish_date"] = fund["report_date"] + pd.Timedelta(days=45)

    # Apply point-in-time lag
    fund = _apply_pit_lag(fund, "publish_date", lag_days)

    logger.info(
        "yfinance financials: {} rows from {} tickers",
        len(fund), fund["ticker"].nunique(),
    )
    return fund


def _safe_get(series: pd.Series, keys: List[str]) -> Optional[float]:
    """Try multiple row labels, return the first non-NaN value found."""
    for k in keys:
        if k in series.index:
            v = series[k]
            if pd.notna(v):
                return float(v)
    return np.nan


# ---------------------------------------------------------------------------
# Point-in-time alignment
# ---------------------------------------------------------------------------

def _apply_pit_lag(
    df: pd.DataFrame,
    publish_col: str,
    lag_days: int,
) -> pd.DataFrame:
    """Shift ``publish_date`` forward by *lag_days* and snap to next trading day.

    Adds a ``date`` column representing the first trading day the data is
    "known" to the market.
    """
    df = df.copy()
    df[publish_col] = pd.to_datetime(df[publish_col])
    df["_avail"] = df[publish_col] + pd.Timedelta(days=lag_days)
    # Snap to next trading day
    df["date"] = df["_avail"].apply(
        lambda d: next_trading_day(d) if pd.notna(d) else pd.NaT
    )
    df.drop(columns=["_avail"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Merge statements into a single table
# ---------------------------------------------------------------------------

def _merge_simfin_statements(
    stmts: Dict[str, pd.DataFrame],
    lag_days: int,
) -> pd.DataFrame:
    """Combine SimFin income / balance / cashflow into one point-in-time table."""
    # SimFin DataFrames typically have MultiIndex (Ticker, Report Date)
    # and a "Publish Date" column.

    combined_frames = []

    # Column mappings: simfin_col → our schema col
    income_map = {
        "Revenue": "revenue",
        "Gross Profit": "gross_profit",
        "Operating Income (Loss)": "operating_income",
        "Net Income": "net_income",
        "Earnings Per Share, Diluted": "eps_diluted",
    }
    balance_map = {
        "Total Assets": "total_assets",
        "Total Liabilities": "total_liabilities",
        "Total Equity": "total_equity",
        "Cash, Cash Equivalents & Short Term Investments": "cash_and_equivalents",
        "Total Debt": "total_debt",
        "Shares (Diluted)": "shares_outstanding",
    }
    cashflow_map = {
        "Net Cash from Operating Activities": "cash_from_operations",
        "Capital Expenditures": "capex",
        "Free Cash Flow": "free_cash_flow",
    }

    mapping_pairs = [
        ("income", income_map),
        ("balance", balance_map),
        ("cashflow", cashflow_map),
    ]

    for stmt_name, col_map in mapping_pairs:
        raw = stmts.get(stmt_name)
        if raw is None or raw.empty:
            continue
        raw = raw.reset_index()
        # Normalise column names to handle SimFin variations
        rename = {}
        for src, dst in col_map.items():
            if src in raw.columns:
                rename[src] = dst
        sub = raw.rename(columns=rename)
        # Keep only mapped columns + identifiers
        keep = ["Ticker", "Report Date", "Publish Date"] + list(rename.values())
        keep = [c for c in keep if c in sub.columns]
        sub = sub[keep].copy()
        sub.rename(columns={"Ticker": "ticker", "Report Date": "report_date",
                            "Publish Date": "publish_date"}, inplace=True)
        combined_frames.append(sub)

    if not combined_frames:
        return pd.DataFrame()

    # Outer-merge on (ticker, report_date)
    merged = combined_frames[0]
    for extra in combined_frames[1:]:
        merged = merged.merge(extra, on=["ticker", "report_date", "publish_date"], how="outer")

    # Apply point-in-time lag
    merged = _apply_pit_lag(merged, "publish_date", lag_days)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_fundamentals(
    tickers: Optional[List[str]] = None,
    prices_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
) -> Path:
    """Full pipeline: fetch → merge → PIT-align → validate → save.

    Returns the output Parquet path.
    """
    cfg = load_config()
    lag_days = cfg["fundamentals"]["publication_lag_days"]
    variant = cfg["fundamentals"]["simfin_variant"]
    api_key = cfg["fundamentals"].get("simfin_api_key") or os.environ.get("SIMFIN_API_KEY")

    # --- Resolve tickers ---
    if tickers is None:
        from quant_platform.core.data.prices import get_ticker_list
        tickers = get_ticker_list(cfg["universe"]["index_membership"])

    # --- Try SimFin first ---
    _setup_simfin(api_key)
    stmts = _fetch_simfin_statements(variant)
    fund = _merge_simfin_statements(stmts, lag_days)

    if fund.empty:
        logger.warning("SimFin returned no data – falling back to yfinance financials")
        fund = _fetch_financials_yf(tickers, lag_days)

    # If yfinance financials also returned nothing, fall back to market-cap only
    if fund.empty:
        logger.warning("yfinance financials also empty – using market-cap-only fallback")
        dates = trading_dates()
        fund = _fetch_market_cap_yf(tickers, dates)

    # --- Filter to our ticker universe ---
    if "ticker" in fund.columns and not fund.empty:
        fund = fund[fund["ticker"].isin(tickers)].copy()

    # --- Enrich with close-price-based market cap if missing ---
    if prices_path is None:
        prices_path = resolve_data_path(cfg["paths"]["prices_dir"], "prices.parquet")
    if Path(prices_path).exists() and not fund.empty:
        from quant_platform.core.utils.io import load_parquet
        px = load_parquet(prices_path, columns=["date", "ticker", "close"])
        px["date"] = pd.to_datetime(px["date"])
        fund["date"] = pd.to_datetime(fund["date"])
        if "shares_outstanding" in fund.columns:
            fund = fund.merge(px, on=["date", "ticker"], how="left", suffixes=("", "_px"))
            mask = fund["market_cap"].isna() & fund["shares_outstanding"].notna()
            fund.loc[mask, "market_cap"] = (
                fund.loc[mask, "shares_outstanding"] * fund.loc[mask, "close"]
            )
            if "close_px" in fund.columns:
                fund.drop(columns=["close_px"], inplace=True, errors="ignore")

    # --- Compute book_value_per_share if possible ---
    if "total_equity" in fund.columns and "shares_outstanding" in fund.columns:
        fund["book_value_per_share"] = fund["total_equity"] / fund["shares_outstanding"]
    else:
        fund["book_value_per_share"] = np.nan

    # --- Fill missing schema columns with appropriate null types ---
    import pyarrow as pa
    for i, field in enumerate(FUNDAMENTALS_SCHEMA):
        if field.name not in fund.columns:
            if field.type in (pa.date32(), pa.date64()):
                fund[field.name] = pd.NaT
            elif field.name in ("date", "ticker"):
                fund[field.name] = None
            else:
                fund[field.name] = np.nan

    fund = fund[[f.name for f in FUNDAMENTALS_SCHEMA]]

    # --- Align to calendar ---
    if "date" in fund.columns and not fund.empty:
        fund["date"] = pd.to_datetime(fund["date"])
        fund = align_to_calendar(fund, date_col="date")

    fund = fund.sort_values(["date", "ticker"]).reset_index(drop=True)

    validate_dataframe(fund, "fundamentals")

    logger.info("Fundamentals: {} rows, {} tickers", len(fund), fund["ticker"].nunique())

    if output_path is None:
        output_path = resolve_data_path(cfg["paths"]["fundamentals_dir"], "fundamentals.parquet")
    return save_parquet(fund, output_path, schema=FUNDAMENTALS_SCHEMA)
